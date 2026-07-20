"""
Run GPC-RANK on a pretrained diffusion policy by rolling out its action proposals in DriftWorld:
At each planning step,
    - the diffusion policy proposes num_trial action trajectories
    - DriftWorld simulates them
    - the candidate with the best predicted reward is chosen to execute
For each evaluation seed, the code saves the full ground-truth PushT rollout, and writes the scores as .npy files.
"""
import os
import logging
import numpy as np
import torch
import yaml
import collections
import torchvision.transforms.v2 as v2
from tqdm.auto import tqdm
import imageio
import time

from eval.util_eval_setup import set_seed
from gpc_rank.reward_predictor import RewardPredictor, estimate_reward_torch
from utils_model import create_model
from gpc_rank.diffusion_policy.utils_model import create_policy
from gpc_rank.diffusion_policy.utils import create_injected_noise, normalize_data, unnormalize_data
from gpc_rank.pusht_env import PushTImageEnv

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


def run_gpc_rank(cfg, num_trial, num_parallel, start_number_test, end_number_test, baseline_render_size = 96):
    """
    Run GPC-RANK on a pretrained diffusion policy by rolling out its action proposals in the world model

    Inputs:
        cfg: hydra cfg
        num_trial: number of trials for GPC-RANK
        num_parallel: number of proposals to run through the world model in parallel
        start_number_test: index of the first test seed to evaluate (inclusive)
        end_number_test: index one past the last test seed to evaluate (exclusive)
        baseline_render_size: resolution size for the ground-truth baseline videos only (display/eval).
            Model inputs and world-model conditioning are unaffected (they stay at the original size).
    Outputs:
        average IoU score achieved across evaluation trials, after applying GPC-RANK to the policy
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    dynamics_stats = {'agent_pos': {'min': np.array([2.0407837e-04, 1.0189312e+00], dtype=np.float32), 'max': np.array([509.08173, 509.43417], dtype=np.float32)}, 'action': {'min': np.array([0., 0.], dtype=np.float32), 'max': np.array([511., 511.], dtype=np.float32)}}
    domain18_stats = {'agent_pos': {'min': np.array([9.897889, 9.63592 ], dtype=np.float32), 'max': np.array([499.517  , 499.00488], dtype=np.float32)}, 'action': {'min': np.array([2., 2.], dtype=np.float32), 'max': np.array([511., 511.], dtype=np.float32)}}

    num_diffusion_iters = cfg.policy.num_diffusion_iters
    pred_horizon = cfg.data.pred_horizon
    obs_horizon = cfg.data.obs_horizon
    action_horizon = cfg.data.action_horizon
    output_dir = f"{cfg.output_dir}/num_trial_{num_trial}_seeds_{start_number_test}_{end_number_test}"
    resize_scale = cfg.data.resize_scale
    action_dim = 2
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Loading diffusion policy from {cfg.ckpt.policy_checkpoint}")
    nets = setup_diff_policy(cfg, cfg.ckpt.policy_checkpoint)

    log.info(f"Loading world model from {cfg.ckpt.world_model_checkpoint}")
    nets["denoiser"] = setup_world_model(cfg, cfg.ckpt.world_model_checkpoint)
    nets = nets.to(device)
    nets.eval()

    K = nets["denoiser"].num_history_frames
    cur_idx = K - 1
    log.info(f"Multi-frame world model: K={K} cur_idx={cur_idx}")

    # ResNet18-based reward predictors that estimate the (x, y, theta) pose of the T-block
    log.info("Loading reward predictors")
    reward_predictor_unnormalized_xy = RewardPredictor().to(device)
    reward_predictor_cossin_angle = RewardPredictor().to(device)

    reward_predictor_unnormalized_xy.load_state_dict(torch.load(cfg.ckpt.reward_predictor_xy_checkpoint))
    reward_predictor_unnormalized_xy.eval()

    reward_predictor_cossin_angle.load_state_dict(torch.load(cfg.ckpt.reward_predictor_angle_checkpoint))
    reward_predictor_cossin_angle.eval()


    log.info("Start GPC-RANK evaluation")
    scores = []
    json_dict = dict()

    env_j_scores = []
    env_seed = 100050   # first test seed
    forward_pass_time_list = []

    with open("./domains_yaml/{}.yml".format('push_t'), 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    env_id = data_loaded["domain_id"]

    json_dict["domain_{}".format(env_id)] = []

    log.info("\nEval Diff Policy on Domain #{}:".format(env_id))

    # start_number_test / end_number_test are passed in so the seed range can be split across GPUs
    env_seed = env_seed + start_number_test

    for test_index in range(start_number_test, end_number_test):
        # Seed per trial rather than once per process
        # That way, the result is not affected by shard boundaries: e.g., doing [0,50) matches doing [0,25) + [25,50)
        set_seed(cfg.train.seed + test_index)
        noise_scheduler = create_injected_noise(num_diffusion_iters)

        # limit environment interaction to cfg.env.max_steps steps before termination
        max_steps = cfg.env.max_steps
        env = PushTImageEnv(domain_filename='push_t', resize_scale=resize_scale)
        env.seed(env_seed)
        # get first observation
        obs, info = env.reset()
        aa = env.goal_pose

        target_pose = torch.tensor(aa, dtype=torch.float32).to(device)
        target_pose[2] = target_pose[2] % (2 * np.pi)
        # keep a queue of last 2 steps of history observations
        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
        # save visualization and rewards
        # draw_action_marker=False keeps the ground-truth rollout free of the red action cross
        baseline_imgs = [env.render_highres(baseline_render_size, draw_action_marker=False)]

        rewards = list()
        done = False
        step_idx = 0

        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(96),
            v2.ToDtype(torch.float32, scale=True),
        ])

        # Model predictive control
        tqdm._instances.clear()

        last_obs_gt = []
        with tqdm(total=max_steps, desc="Eval Trial #{}".format(test_index)) as pbar:
            while not done:
                #### Process last few images and agent positions to form observation vector obs_cond
                B = 1
                # stack the last obs_horizon number of observations
                images = np.stack([x['image'] for x in obs_deque])
                agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

                # normalize observation
                nagent_poses = normalize_data(agent_poses, stats=domain18_stats['agent_pos'])
                nagent_poses_dynamics = normalize_data(agent_poses, stats=dynamics_stats['agent_pos'])

                # device transfer
                nimages = torch.from_numpy(images).to(device, dtype=torch.float32) # (2,3,96,96)
                nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
                nagent_poses_dynamics = torch.from_numpy(nagent_poses_dynamics).to(device, dtype=torch.float32)

                #### In the loop below:
                #### (1) diffusion policy starts from random noise and generates action trajectory
                ####     Repeat num_trial times to get num_trial candidate trajectories
                #### (2) Then world model simulates what the environment looks like if these actions are taken
                with torch.no_grad():
                    # visual feature representations of nimages (the normalized context images, i.e. the recent history)
                    image_features = nets["vision_encoder"](nimages)
                    image_features = image_features.squeeze()

                    # concat with low-dim observations, which are the agent positions
                    obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                    # num_trial sequences of random Gaussian noise
                    naction = torch.randn((num_trial, pred_horizon, action_dim), device=device)

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    # DDPM-style diffusion
                    for k in noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = nets["invariant"](
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond.repeat(num_trial, 1)
                        )

                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                    # unnormalize action
                    naction = naction.detach().to('cpu').numpy()
                    # (B, pred_horizon, action_dim)
                    action_pred = unnormalize_data(naction, stats=domain18_stats['action'])

                    # only take action_horizon number of actions
                    start = obs_horizon - 1
                    end = start + action_horizon
                    action = action_pred[:, start:,:] # (action_horizon, action_dim)

                    pred_imgs = last_obs_gt
                    last_obs_gt = []

                    action = np.swapaxes(action,0,1) # new shape (time, num_trials, action_dim)
                    action_mean = np.mean(action, axis = 1) # shape (time, action_dim)
                        # action_mean is the average action taken over all num_trial trials

                    action_mean = np.expand_dims(action_mean, axis=1).repeat(num_trial, axis=1)
                        # shape (time, num_trials, action_dim)
                        # duplicates the mean action num_trial times to match the original shape

                    # exploration trick: scales the difference between each individual action and the mean action by a factor of 1.01.
                    # This artificially spreads out the num_trial candidate trajectories slightly, increasing diversity.
                    action = action_mean + 1.01 * (action - action_mean)
                    action = np.swapaxes(action,0,1) # new shape: (num_trials, time, action_dim)

                    if len(pred_imgs) > 0:
                        # seed window: last K frames s_(t-cur_idx), ..., s_t. Left-pad by repeating the
                        # earliest available frame if fewer than K frames have accumulated yet.
                        if pred_imgs.shape[1] < K:
                            pad = np.repeat(pred_imgs[:, :1], K - pred_imgs.shape[1], axis=1)
                            cur_state = np.concatenate([pad, pred_imgs], axis=1)[:, -K:]
                        else:
                            cur_state = pred_imgs[:, -K:] # current+history states, shape (num_trial, K, 3, 96, 96)
                        denoiser_input_action = action.copy() # shape (num_trials, time, action_dim)
                        denoiser_input_action = normalize_data(denoiser_input_action, stats=dynamics_stats['action'])

                        denoiser_input_action = torch.tensor(denoiser_input_action, dtype=torch.float32).to(device)
                        cur_state = torch.tensor(cur_state, dtype=torch.float32).to(device)

                        if cfg.data.normalize_img:
                            cur_state = (cur_state - 0.5) / 0.5

                        # prepend cur_idx offset actions so the candidate future actions land at
                        # denoiser_input_action[:, cur_idx:] (the first cur_idx entries are unused offset).
                        if cur_idx > 0:
                            pad_act = torch.zeros((denoiser_input_action.shape[0], cur_idx, action_dim), device=device)
                            denoiser_input_action = torch.cat([pad_act, denoiser_input_action], dim=1)

                        # NOTE WORLD MODEL SIMULATION
                        log.info(f"call multi-history world model with {num_trial} trials in chunks of {num_parallel}")
                        start_time = time.perf_counter()

                        pred_images_chunks = []
                        for start_idx in range(0, num_trial, num_parallel):
                            end_idx = min(start_idx + num_parallel, num_trial)
                            cur_state_chunk = cur_state[start_idx:end_idx]
                            action_chunk = denoiser_input_action[start_idx:end_idx]

                            pred_images_chunk = nets["denoiser"].sample_autoregressive(
                                cur_state = cur_state_chunk,
                                actions = action_chunk,
                            )
                            pred_images_chunks.append(pred_images_chunk)

                        pred_images = torch.cat(pred_images_chunks, dim=0)

                        end_time = time.perf_counter()
                        forward_pass_time = end_time - start_time
                        if test_index != start_number_test:
                            forward_pass_time_list.append(forward_pass_time)
                            log.info(f"(demo {test_index}/{end_number_test-start_number_test}) call multi-history world model: {forward_pass_time} | running average {np.mean(forward_pass_time_list):.6f}")
                        else:
                            log.info(f"(demo {test_index}/{end_number_test-start_number_test}) call multi-history world model: {forward_pass_time}")

                        if cfg.data.normalize_img:
                            pred_images = (pred_images * 0.5) + 0.5

                        pred_images = pred_images.detach().cpu().numpy()
                        # pred_images is (num_trial, K + time, 3, 96, 96); drop the K seed frames.
                        pred_imgs = np.concatenate((pred_imgs, pred_images[:, K:]), axis=1)

                    #### NOTE GPC-RANK: evaluate and rank the num_trial candidate action trajectories that were simulated by the world model
                    all_reward_candidate = [] # list to store the num_trial calculated rewards
                    if len(pred_imgs) > 0:
                        last_pred_image = torch.tensor(pred_imgs[:,-1], dtype=torch.float32, device=device)
                            # shape (num_trial, 3, 96, 96), since it contains the final frame in each of the num_trial simulated video sequences

                        for i in range(num_trial):
                            unnormalized_xy = reward_predictor_unnormalized_xy(torch.unsqueeze(last_pred_image[i], 0))[0]
                                # unnormalized_xy: predicted (x,y) coordinates of the T block
                            cossin_angle = reward_predictor_cossin_angle(torch.unsqueeze(last_pred_image[i], 0))[0]
                                # cossin_angle: predicted (cos(theta), sin(theta)) pair that represents the block's orientation

                            cossin_angle = cossin_angle/torch.sqrt(cossin_angle[0]*cossin_angle[0] + cossin_angle[1]*cossin_angle[1])
                            block_angle = torch.atan2(cossin_angle[1], cossin_angle[0]) % (2 * torch.pi) # recovers theta

                            block_pose = torch.stack((unnormalized_xy[0], unnormalized_xy[1], block_angle))
                            reward = estimate_reward_torch(block_pose, target_pose)
                                # finds reward, which computes the sum of distances between the vertices of the predicted block and the target block
                                # smaller value is better
                            all_reward_candidate.append(reward.detach().cpu().numpy())

                    if len(all_reward_candidate) > 0: # RANKING STEP
                        pick_index = np.argsort(all_reward_candidate, kind="stable")[0]
                        action_pick = action[pick_index][:end] # best sequence
                        log.info(all_reward_candidate[pick_index])
                    else:
                        action_pick = action[0][:end]

                #### NOTE Run the best action sequence (action_pick) in the real push-T env
                for i in range(len(action_pick)):
                    # stepping env
                    obs, reward, done, _, info = env.step(action_pick[i])
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)
                    baseline_imgs.append(env.render_highres(baseline_render_size, draw_action_marker=False))

                    # use a marker-free render for the seed frames
                    # this keeps both the saved video and the world-model conditioning marker-free
                    clean_frame = env._render_frame(mode='rgb_array')
                    last_obs_gt.append(np.expand_dims(transform(clean_frame).numpy(), axis=0))

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix({"current": reward, "max": max(rewards)})
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break

                # Prepare the "history" of the world model's next simulation
                last_obs_gt = np.array(last_obs_gt)
                last_obs_gt = np.transpose(last_obs_gt, (1, 0, 2, 3, 4))
                    # new shape: (Batch, Time, Channels, Height, Width)
                last_obs_gt = np.tile(last_obs_gt, (num_trial, 1, 1, 1, 1))
                    # duplicates it num_trial times

        env_seed += 1
        max_reward = max(rewards)
        env_j_scores.append(max_reward)
        log.info(f"(demo {test_index}/{end_number_test-start_number_test}) reward {max_reward} | running average {np.mean(env_j_scores)}")

        log.info(f"Saving visualization of the first few demos")
        log.info(f"imgs: {len(baseline_imgs)}") # number of (size, size, 3) uint8 RGB frames
        imageio.mimsave(f"{output_dir}/baseline_single_dp_on_domain_{env_id}_test_{test_index}_res{baseline_render_size}.mp4",
                        baseline_imgs, fps=4)
        np.save(f"{output_dir}/corrected_sampling_based_testing_no_simulation_planning_receding_result_from_index_f{start_number_test}.npy", np.array(env_j_scores))

    answer = np.mean(env_j_scores)
    log.info("Single DP on Domain #{} Avg Score: {}".format(env_id, answer))
    scores.append(env_j_scores)
    np.save(f"{output_dir}/final_corrected_sampling_based_testing_no_simulation_planning_receding_result_from_index_f{start_number_test}.npy", np.array(scores))
    return answer
