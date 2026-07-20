"""
Training loop for diffusion policy
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.set_float32_matmul_precision('high')
import numpy as np
import wandb
import logging
import time
import random
from collections import defaultdict
import json
import torch.nn as nn
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from utils import PushTImageDataset, create_injected_noise
from models import get_resnet, replace_bn_with_gn, ConditionalUnet1D

log = logging.getLogger(__name__)

def set_seed(seed):
    if seed == -1:
        seed = int(time.time())
    log.info(f"Seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed

def train(cfg):
    """
    Train model, given hydra config cfg
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    set_seed(cfg.train.seed)

    log.info("Creating dataloader")
    dataset = PushTImageDataset(
        dataset_path=cfg.data.dataset_path_dir,
        pred_horizon=cfg.data.pred_horizon,
        obs_horizon=cfg.data.obs_horizon,
        action_horizon=cfg.data.action_horizon,
        id=0,
        num_demos=cfg.data.num_train_demos,
        resize_scale=cfg.data.resize_scale,
        pretrained=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=True,
        pin_memory=cfg.dataloader.pin_memory,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        persistent_workers=True
    )

    log.info("Creating model")
    nets = nn.ModuleDict({})

    vision_encoder = get_resnet()
    vision_encoder = replace_bn_with_gn(vision_encoder)
    nets['vision_encoder'] = vision_encoder

    vision_feature_dim = 512
    lowdim_obs_dim = 2
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2

    invariant = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * cfg.data.obs_horizon
    )
    nets['invariant'] = invariant 
    nets = nets.to(device)

    noise_scheduler = create_injected_noise(cfg.model.num_diffusion_iters)
    
    ema = EMAModel(
        parameters=nets.parameters(),
        power=cfg.train.decay
    )
    
    log.info("Creating optimizer")
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=cfg.opt.lr,
        weight_decay=cfg.opt.weight_decay
    )
    
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=cfg.opt.num_warmup_steps,
        num_training_steps=len(dataloader) * cfg.train.num_epochs
    )

    actual_step = 0 # current step
    to_skip = False # Whether to skip to the correct location in dataloader

    log.info("Creating model / Restoring checkpoint")
    os.makedirs(f"{cfg.output_dir}/ckpt_save", exist_ok=True)  
    if os.path.exists(cfg.path_ckpt_latest):
        ckpt = torch.load(cfg.path_ckpt_latest, weights_only=False)
        nets.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['scheduler'])
        actual_step = ckpt['step'] + 1
        del ckpt
        log.info(f"Restored from step {actual_step} ckpt")
        if actual_step % len(dataloader) != 0:
            to_skip = True

    # Set up wandb run
    wandb.login(key=cfg.wandb_info.key)
    if not os.path.exists(cfg.wandb_info.saved_run_id):
        run = wandb.init(
            entity=cfg.wandb_info.entity,
            project=cfg.wandb_info.project,
            name=cfg.wandb_info.name,
            dir=cfg.output_dir
        )
        run_id = run.id
        with open(cfg.wandb_info.saved_run_id, 'w') as f:
            json.dump({'run_id': run_id}, f)
        log.info(f"Started new wandb run {run_id}")
    else:
        log.info(f"Resuming wandb run")
        with open(cfg.wandb_info.saved_run_id, 'r') as f:
            run_id = json.load(f)['run_id']
        run = wandb.init(
            entity=cfg.wandb_info.entity,
            project=cfg.wandb_info.project,
            id=run_id,
            resume="allow",
            dir=cfg.output_dir
        )

    n_params = sum(p.numel() for p in nets.parameters() if p.requires_grad)
    wandb.log({'num_params': n_params, 'seed': cfg.train.seed}, step=0)

    start_ep = actual_step // len(dataloader) + 1
    for epoch_idx in range(start_ep, cfg.train.num_epochs + 1):
        log.info(f"(epoch {epoch_idx}) start")
        total_ep = defaultdict(float)
        
        for batch_idx, nbatch in enumerate(dataloader):
            if to_skip:
                cur_step = len(dataloader) * (epoch_idx - 1) + batch_idx
                if cur_step < actual_step:
                    continue
                elif cur_step == actual_step:
                    to_skip = False

            nimage = nbatch['image'][:, :cfg.data.obs_horizon].to(device)
            nagent_pos = nbatch['agent_pos'][:, :cfg.data.obs_horizon].to(device)
            naction = nbatch['action'].to(device)
            B = nagent_pos.shape[0]

            # encoder vision features
            image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))
            image_features = image_features.reshape(*nimage.shape[:2], -1)

            # concatenate vision feature and low-dim obs
            obs_features = torch.cat([image_features, nagent_pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1) # (B, obs_horizon * obs_dim)

            # sample noises to add to actions
            noise = torch.randn(naction.shape, device=device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            if epoch_idx == start_ep and batch_idx == 0:
                log.info(f"Shapes at epoch {start_ep} batch 0:")
                log.info(f"  nimage: {nimage.shape} | {nimage.min()} | {nimage.max()}")
                log.info(f"  nagent_pos: {nagent_pos.shape}")
                log.info(f"  naction: {naction.shape}")
                log.info(f"  vision_encoder input: {nimage.flatten(end_dim=1).shape}")
                log.info(f"  invariant (unet) input 'noisy_actions': {naction.shape}")
                log.info(f"  invariant (unet) input 'timesteps': {timesteps.shape}")
                log.info(f"  invariant (unet) input 'global_cond' (obs_cond): {obs_cond.shape}")

            # add noise to the clean images according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

            # predict the noise residual
            noise_pred = nets["invariant"](noisy_actions, timesteps, global_cond=obs_cond)

            loss = nn.functional.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            lr_scheduler.step()
            ema.step(nets.parameters())

            metrics = {
                'loss': loss.item(),
                'lr': lr_scheduler.get_last_lr()[0]
            }
            wandb.log(metrics, step=actual_step)
            
            if batch_idx % 100 == 0:
                log.info(f"(epoch {epoch_idx}) (batch {batch_idx}/{len(dataloader)}) loss: {metrics['loss']:.4f}")

            for k, v in metrics.items():
                if k != 'lr': 
                    total_ep[k] += v

            if actual_step % cfg.train.ckpt_every == 0 and actual_step > 0:
                log.info(f"(epoch {epoch_idx}) Saving latest ckpt at step {actual_step}")
                if os.path.exists(cfg.path_ckpt_latest):
                    try:
                        os.replace(cfg.path_ckpt_latest, cfg.path_ckpt_2nd_latest)
                    except Exception as e:
                        log.info(f"Failed to move latest checkpoint to 2nd latest: {e}")
                checkpoint = {
                    'step': actual_step,
                    'model': nets.state_dict(),
                    'ema': ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'cfg': cfg
                }
                torch.save(checkpoint, cfg.path_ckpt_latest)        
                
            actual_step += 1

        if epoch_idx >= cfg.train.ckpt_save_min and epoch_idx % cfg.train.ckpt_save == 0:
            log.info(f"(epoch {epoch_idx}) Saving latest ckpt at epoch {epoch_idx} in save folder")
            checkpoint = {
                'step': actual_step,
                'model': nets.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'cfg': cfg
            }
            torch.save(checkpoint, f"{cfg.output_dir}/ckpt_save/ckpt-ep{epoch_idx}.pth")
            

        average_ep = {f"ep_{key}": total / len(dataloader) for key, total in total_ep.items()}
        wandb.log(average_ep, step=actual_step)