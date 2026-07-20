"""
Compute IoU of the diffusion policy when rolled out in DriftWorld, versus in the ground-truth simulator.

Each trial runs two closed-loop rollouts side by side:
    - simulated: diffusion policy generates an action chunk,
      DriftWorld simulates the next visual states,
      the reward predictor extracts the block pose, and IoU is computed from it
    - real: the same policy generates an action chunk, which is executed in PushTImageEnv

Both rollouts begin with the same initial frame.
"""
import os
import logging
import numpy as np
import torch
import yaml
import collections
from tqdm.auto import tqdm
import imageio

from eval.util_eval_setup import set_seed
from gpc_rank.reward_predictor import RewardPredictor
from utils_model import create_model
from gpc_rank.diffusion_policy.utils_model import create_policy
from gpc_rank.diffusion_policy.utils import create_injected_noise, normalize_data, unnormalize_data
from gpc_rank.pusht_env import PushTImageEnv, pymunk_to_shapely

log = logging.getLogger(__name__)

def setup_world_model(cfg, filepath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("Creating model")
    denoiser = create_model(cfg, device)
    log.info("Restoring ckpt")

    if os.path.exists(filepath):
        ckpt = torch.load(filepath, weights_only=False)
        denoiser.load_state_dict(ckpt['model'])
        actual_step = ckpt['step']
        del ckpt
        log.info(f"Restored from step {actual_step} ckpt")
        return denoiser
    else:
        log.info(f"Checkpoint {filepath} does not exist")
        return

def setup_diff_policy(cfg, filepath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("Creating diffusion policy")
    nets = create_policy(cfg, filepath, device)
    log.info("Creating diffusion policy: done")
    return nets

def process_frame(pred_image_np, value_range):
    """
    Args:
        pred_image_np: (C, H, W) numpy array of generated frame from world model (float32)
        value_range: tuple (0, 1) or (-1, 1)
    Returns:
        (H, W, C) numpy array of processed frame (uint8)
    """
    res = pred_image_np.transpose(1, 2, 0)
    res = np.clip(res, a_min=value_range[0], a_max=value_range[1])
    res = (res - value_range[0]) / (value_range[1] - value_range[0]) # now all values of res are floats in [0, 1]
    res = (res * 255).astype(np.uint8) # now all values are ints in [0, 255]
    return res

def build_obs_cond(nets, obs_deques, B, obs_horizon, domain18_stats, device):
    """
    Create the policy's conditioning vector from B observation deques.
    Args:
        obs_deques: list of B deques, each holding obs_horizon dicts with
                    'image' (3, 96, 96) in [0, 1] and 'agent_pos' (2,)
    Returns:
        (B, obs_horizon*feat) tensor
    """
    images = np.stack([np.stack([x['image'] for x in obs_deques[b]]) for b in range(B)])
    agent_poses = np.stack([np.stack([x['agent_pos'] for x in obs_deques[b]]) for b in range(B)])
    nagent_poses = normalize_data(agent_poses, stats=domain18_stats['agent_pos'])

    nimages = torch.from_numpy(images).to(device, dtype=torch.float32)
    nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)

    # vision encoder expects (N, 3, 96, 96); flatten batch & horizon, then restore
    nimages_flat = nimages.reshape(B * obs_horizon, *nimages.shape[2:])
    image_features = nets["vision_encoder"](nimages_flat)        # (B*obs_horizon, 512)
    image_features = image_features.reshape(B, obs_horizon, -1)  # (B, obs_horizon, 512)

    obs_features = torch.cat([image_features, nagent_poses], dim=-1)  # (B, obs_horizon, feat)
    return obs_features.flatten(start_dim=1)                          # (B, obs_horizon*feat)

def denoise_action_chunk(nets, noise_scheduler, obs_cond, naction_init, plan_seed,
                         num_diffusion_iters, obs_horizon, action_horizon,
                         domain18_stats, device):
    """
    Sample one action chunk with DDPM.
    Returns:
        (B, action_horizon, 2) numpy array of actions
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(plan_seed)

    naction = naction_init.clone()
    noise_scheduler.set_timesteps(num_diffusion_iters)
    for k in noise_scheduler.timesteps:
        noise_pred = nets["invariant"](sample=naction, timestep=k, global_cond=obs_cond)
        naction = noise_scheduler.step(model_output=noise_pred, timestep=k,
                                       sample=naction, generator=gen).prev_sample

    action_pred = unnormalize_data(naction.detach().cpu().numpy(), stats=domain18_stats['action'])
    start = obs_horizon - 1
    return action_pred[:, start:start + action_horizon, :]

def eval_policy_simulated(cfg, policy_ckpt, epoch, num_parallel=100):
    """
    Evaluates the diffusion policy twice per trial: once rolled out in DriftWorld, once in the real
    PushT environment. Runs `num_parallel` test trials concurrently as a batch.

    Both rollouts are closed-loop on their own observations but share their diffusion sampling noise,
    so a trial's two scores form a matched pair.

    Inputs:
        cfg: hydra cfg
        policy_ckpt: filepath to diffusion policy checkpoint
        epoch: epoch of the above policy ckpt
        num_parallel: number of test trials (seeds) to simulate concurrently in one batch
    Outputs:
        tuple of average IoU in DriftWorld, and average IoU in ground-truth
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    dynamics_stats = {
        'action': {'min': np.array([0., 0.], dtype=np.float32), 'max': np.array([511., 511.], dtype=np.float32)}
    }
    domain18_stats = {
        'agent_pos': {'min': np.array([9.897889, 9.63592 ], dtype=np.float32), 'max': np.array([499.517  , 499.00488], dtype=np.float32)},
        'action': {'min': np.array([2., 2.], dtype=np.float32), 'max': np.array([511., 511.], dtype=np.float32)}
    }

    num_diffusion_iters = cfg.policy.num_diffusion_iters
    pred_horizon = cfg.data.pred_horizon
    obs_horizon = cfg.data.obs_horizon
    action_horizon = cfg.data.action_horizon
    output_dir = f"{cfg.output_dir}/epoch_{epoch}"
    resize_scale = cfg.data.resize_scale
    action_dim = 2

    log.info(f"Loading diffusion policy from {policy_ckpt}")
    nets = setup_diff_policy(cfg, policy_ckpt)

    log.info(f"Loading world model from {cfg.ckpt.world_model_checkpoint}")
    nets["denoiser"] = setup_world_model(cfg, cfg.ckpt.world_model_checkpoint)
    nets = nets.to(device)
    nets.eval()

    # Multi-frame model history
    K = nets["denoiser"].num_history_frames
    cur_idx = K - 1
    log.info(f"Multi-history world model: K={K} cur_idx={cur_idx}")

    log.info("Loading reward predictors")
    reward_predictor_unnormalized_xy = RewardPredictor().to(device)
    reward_predictor_cossin_angle = RewardPredictor().to(device)
    reward_predictor_unnormalized_xy.load_state_dict(torch.load(cfg.ckpt.reward_predictor_xy_checkpoint))
    reward_predictor_cossin_angle.load_state_dict(torch.load(cfg.ckpt.reward_predictor_angle_checkpoint))
    reward_predictor_unnormalized_xy.eval()
    reward_predictor_cossin_angle.eval()

    log.info(f"Start diffusion policy evaluation in DriftWorld + real env (num_parallel={num_parallel})")
    scores_sim = []
    scores_real = []
    base_env_seed = 100000   # first test seed

    with open("./domains_yaml/{}.yml".format('push_t'), 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    env_id = data_loaded["domain_id"]

    log.info("\nEval Diff Policy in DriftWorld vs real on Domain #{}:".format(env_id))

    start_number_test = 0
    end_number_test = start_number_test + 100

    value_range = (-1,1) if cfg.data.normalize_img else (0,1)
    max_steps = cfg.env.max_steps
    os.makedirs(output_dir, exist_ok=True)

    for batch_start in range(start_number_test, end_number_test, num_parallel):
        batch_indices = list(range(batch_start, min(batch_start + num_parallel, end_number_test)))
        B = len(batch_indices)   # last batch may be smaller than num_parallel

        # Seed per batch rather than once per process, so
        # each epoch in main_iou.py's ep_list gets an identical RNG stream
        set_seed(cfg.train.seed + batch_start)
        noise_scheduler = create_injected_noise(num_diffusion_iters)

        # Per-trial state (indexed 0..B-1)
        sim_envs = []           # scratch env: block pose overwritten from the reward predictor
        real_envs = []          # actually stepped by the real rollout
        obs_deques_sim = []
        obs_deques_real = []
        wm_seed = []            # world model seed window: last K frames in the MODEL's value range
        wm_img_all = []         # generated frames for the output video
        real_img_all = []       # real env frames for the output video
        rewards_sim = []
        rewards_real = []

        for b, test_index in enumerate(batch_indices):
            # Both envs get the same seed
            sim_env = PushTImageEnv(domain_filename='push_t', resize_scale=resize_scale)
            sim_env.seed(base_env_seed + test_index)
            obs, _ = sim_env.reset()

            real_env = PushTImageEnv(domain_filename='push_t', resize_scale=resize_scale)
            real_env.seed(base_env_seed + test_index)
            real_env.reset()

            sim_envs.append(sim_env)
            real_envs.append(real_env)

            # Same history
            obs_deques_sim.append(collections.deque([obs] * obs_horizon, maxlen=obs_horizon))
            obs_deques_real.append(collections.deque([obs] * obs_horizon, maxlen=obs_horizon))
            
            frame0 = obs['image']  # (3, 96, 96) in [0, 1]
            frame0_model = (frame0 - 0.5) / 0.5 if cfg.data.normalize_img else frame0
            wm_seed.append(collections.deque([frame0_model] * K, maxlen=K))
            wm_img_all.append([process_frame(frame0_model, value_range)])
            real_img_all.append([real_env._render_frame(mode='rgb_array')])
            rewards_sim.append(list())
            rewards_real.append(list())

        done = False
        step_idx = 0
        plan_step = 0

        tqdm._instances.clear()
        desc = "Eval Trials #{}-{}".format(batch_indices[0], batch_indices[-1])
        with tqdm(total=max_steps, desc=desc) as pbar:
            while not done:
                # 1. Action prediction, once per branch. Both share the same start noise and the same
                #    generator seed, so the two chunks differ ONLY through their conditioning.
                plan_seed = cfg.train.seed + batch_start * 100003 + plan_step
                naction_init = torch.randn((B, pred_horizon, action_dim), device=device)

                with torch.no_grad():
                    obs_cond_real = build_obs_cond(
                        nets, obs_deques_real, B, obs_horizon, domain18_stats, device)
                    obs_cond_sim = build_obs_cond(
                        nets, obs_deques_sim, B, obs_horizon, domain18_stats, device)

                    action_real = denoise_action_chunk(
                        nets, noise_scheduler, obs_cond_real, naction_init, plan_seed,
                        num_diffusion_iters, obs_horizon, action_horizon, domain18_stats, device)
                    action_sim = denoise_action_chunk(
                        nets, noise_scheduler, obs_cond_sim, naction_init, plan_seed,
                        num_diffusion_iters, obs_horizon, action_horizon, domain18_stats, device)

                if plan_step == 0:
                    # At the first planning step both branches see identical observations, so with
                    # shared noise the chunks must match bitwise. Guards the pairing wiring.
                    assert np.array_equal(action_real, action_sim), \
                        "paired sampling broken: action chunks differ at plan_step 0"
                plan_step += 1

                # 2. Roll out in the world model.
                #    One sample_autoregressive call simulates the entire action_sim chunk for all B trials at once.
                P = action_sim.shape[1]  # number of future frames to simulate this step (== action_horizon)
                cur_state = torch.tensor(
                    np.stack([np.stack(wm_seed[b]) for b in range(B)]),
                    dtype=torch.float32, device=device)                 # (B, K, 3, 96, 96) in model range

                acts = normalize_data(action_sim, stats=dynamics_stats['action'])  # (B, P, 2)
                acts = torch.tensor(acts, dtype=torch.float32, device=device)
                if cur_idx > 0:
                    # prepend cur_idx offset actions so the future actions land at acts[:, cur_idx:]
                    acts = torch.cat([torch.zeros((B, cur_idx, action_dim), device=device), acts], dim=1)

                with torch.no_grad():
                    pred = nets["denoiser"].sample_autoregressive(
                        cur_state=cur_state, actions=acts)  # (B, K+P, 3, 96, 96)
                new_frames = pred[:, K:]  # (B, P, 3, 96, 96) in model range

                for i in range(P):
                    act = action_sim[:, i]             # (B, 2)
                    frame_model = new_frames[:, i]     # (B, 3, 96, 96) in model range
                    # reward predictors / policy operate in [0, 1]
                    frame01 = (frame_model * 0.5 + 0.5) if cfg.data.normalize_img else frame_model

                    with torch.no_grad():
                        unnormalized_xy = reward_predictor_unnormalized_xy(frame01)  # (B, 2)
                        cossin_angle = reward_predictor_cossin_angle(frame01)        # (B, 2)

                    frame_model_np = frame_model.detach().cpu().numpy()  # (B, 3, 96, 96)
                    frame01_np = frame01.detach().cpu().numpy()          # (B, 3, 96, 96)

                    for b in range(B):
                        # ---- simulated branch ----
                        # Maintain world-model seed (model range) and policy context (in [0, 1]).
                        # NOTE the commanded action stands in for agent_pos here; the real branch below
                        # uses the true PD-controlled position, so the two contexts differ slightly even
                        # when the world model is accurate.
                        wm_seed[b].append(frame_model_np[b])
                        obs_deques_sim[b].append({'image': frame01_np[b], 'agent_pos': act[b]})

                        # For saving the video
                        wm_img_all[b].append(process_frame(frame_model_np[b], value_range))  # (H, W, C)

                        # Calculate physical pose logic
                        cs = cossin_angle[b]
                        cs = cs / torch.sqrt(cs[0]**2 + cs[1]**2)
                        block_angle = torch.atan2(cs[1], cs[0]) % (2 * torch.pi)

                        # Project predicted attributes to the per-trial env to obtain IoU mathematically
                        sim_envs[b].block.position = (unnormalized_xy[b][0].item(), unnormalized_xy[b][1].item())
                        sim_envs[b].block.angle = block_angle.item()

                        goal_body = sim_envs[b]._get_goal_pose_body(sim_envs[b].goal_pose)
                        goal_geom = pymunk_to_shapely(goal_body, sim_envs[b].block.shapes)
                        block_geom = pymunk_to_shapely(sim_envs[b].block, sim_envs[b].block.shapes)

                        coverage = goal_geom.intersection(block_geom).area / goal_geom.area
                        reward = np.clip(coverage / sim_envs[b].success_threshold, 0, 1)
                        rewards_sim[b].append(reward)

                        # ---- real branch: same policy, its own observations ----
                        # env's done flag is ignored so both branches run the full max_steps and stay
                        # step-aligned; max(rewards) is unaffected since reward saturates at 1.0.
                        obs_real, reward_real, _, _, _ = real_envs[b].step(action_real[b, i])
                        obs_deques_real[b].append(obs_real)
                        rewards_real[b].append(reward_real)
                        # marker-free render, to match the world model frames visually
                        real_img_all[b].append(real_envs[b]._render_frame(mode='rgb_array'))

                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "sim": np.mean([max(rewards_sim[b]) for b in range(B)]),
                        "real": np.mean([max(rewards_real[b]) for b in range(B)]),
                    })
                    if step_idx > max_steps: done = True
                    if done: break

        # Record results and save per-trial visualizations
        for b, test_index in enumerate(batch_indices):
            sim_max = max(rewards_sim[b]) if len(rewards_sim[b]) > 0 else 0.0
            real_max = max(rewards_real[b]) if len(rewards_real[b]) > 0 else 0.0
            scores_sim.append(sim_max)
            scores_real.append(real_max)

            log.info(f"demo #{test_index}: DriftWorld IoU {sim_max:.4f} | real IoU {real_max:.4f}")
            # side-by-side [real | world model], each 96x96 -> 96x192
            side_by_side = [np.concatenate([r, w], axis=1)
                            for r, w in zip(real_img_all[b], wm_img_all[b])]
            imageio.mimsave(
                f"{output_dir}/paired_dp_domain_{env_id}_test_{test_index}_len{len(side_by_side)}.mp4",
                side_by_side, fps=10)

        np.save(f"{output_dir}/iou_simulated.npy", np.array(scores_sim))
        np.save(f"{output_dir}/iou_real_paired.npy", np.array(scores_real))

    avg_sim = np.mean(scores_sim)
    avg_real = np.mean(scores_real)
    log.info("Domain #{} Avg IoU -- DriftWorld: {:.4f} | real: {:.4f}".format(env_id, avg_sim, avg_real))
    np.save(f"{output_dir}/iou_simulated.npy", np.array(scores_sim))
    np.save(f"{output_dir}/iou_real_paired.npy", np.array(scores_real))

    return avg_sim, avg_real
