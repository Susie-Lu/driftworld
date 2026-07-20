"""
Training loop for DriftWorld on Push-T
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.set_float32_matmul_precision('high')
import numpy as np
import wandb
import logging
import random
import json

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data.pushT_dataloader import get_pushT_loader
from utils_model import create_model

log = logging.getLogger(__name__)

def ddp_setup():
    """
    Initialize torch.distributed from env vars set by torchrun.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def is_main():
    return int(os.environ.get("RANK", 0)) == 0


def barrier(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


def set_seed(seed):
    if is_main():
        log.info(f"Seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed

def train(cfg):
    """
    Train model, given hydra config cfg. Multi-GPU via torchrun + DDP.
    """
    local_rank, rank, world_size = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main():
        log.info(f"Using device: {device} | world_size={world_size}")

    set_seed(cfg.train.seed)

    if is_main():
        log.info("Creating dataloader")
    dataloader = get_pushT_loader(cfg, rank=rank, world_size=world_size)

    if is_main():
        log.info("Creating model")
    denoiser = create_model(cfg, device)

    if world_size > 1:
        denoiser = DDP(
            denoiser,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        inner = denoiser.module
    else:
        inner = denoiser

    if is_main():
        log.info("Creating optimizer")
    optimizer = torch.optim.AdamW(
        params=inner.parameters(),
        lr=cfg.opt.lr,
        betas=(0.9, cfg.opt.beta2),
        weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6/cfg.opt.lr,
        end_factor=1.0,
        total_iters=cfg.opt.warmup_steps
    )

    actual_step = 0 # current step
    to_skip = False # Whether to skip to the correct location in dataloader

    if is_main():
        log.info("Creating model / Restoring checkpoint")
        os.makedirs(f"{cfg.output_dir}/ckpt_save", exist_ok=True)
    barrier(world_size)

    # Load checkpoint
    if os.path.exists(cfg.path_ckpt_latest):
        ckpt = torch.load(cfg.path_ckpt_latest, map_location="cpu", weights_only=False)
        inner.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        actual_step = ckpt['step'] + 1
        del ckpt
        if is_main():
            log.info(f"Restored from step {actual_step} ckpt")
        if actual_step % len(dataloader) != 0:
            to_skip = True
    elif cfg.model.is_phase_2 == True and os.path.exists(cfg.path_ckpt_phase1):
        # Just started phase 2, so load from phase 1 checkpoint
        ckpt = torch.load(cfg.path_ckpt_phase1, map_location="cpu", weights_only=False)
        inner.load_state_dict(ckpt['model'])
        if is_main():
            log.info(f"Restored from phase 1's step {ckpt['step']} ckpt to begin phase 2 training")
        del ckpt
    barrier(world_size)

    # Set up wandb run
    if is_main():
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

        n_params = sum(p.numel() for p in inner.parameters() if p.requires_grad)
        wandb.log({'num_params': n_params, 'seed': cfg.train.seed}, step=0)
    barrier(world_size)

    start_ep = actual_step // len(dataloader) + 1
    first_pass = True
    for epoch_idx in range(start_ep, cfg.train.num_epochs + 1):
        if world_size > 1 and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch_idx)

        if is_main():
            log.info(f"(epoch {epoch_idx}) start")
        for batch_idx, nbatch in enumerate(dataloader):
            # Reach actual_step by skipping forward in dataloader
            if to_skip:
                cur_step = len(dataloader) * (epoch_idx - 1) + batch_idx
                if cur_step < actual_step:
                    continue
                elif cur_step == actual_step:
                    to_skip = False

            if cfg.data.normalize_img:
                nbatch['image'] = (nbatch['image'] - 0.5) / 0.5 # to [-1,1] range
            if first_pass and is_main():
                log.info(f"batch[image]: {nbatch['image'].shape} | {nbatch['image'].min()} | {nbatch['image'].max()}")
                log.info(f"batch[action]: {nbatch['action'].shape} | {nbatch['action'].min()} | {nbatch['action'].max()}")
            first_pass = False

            loss, metrics = denoiser(nbatch, device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(inner.parameters(), max_norm=cfg.opt.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            inner.update_ema()

            metrics['lr'] = scheduler.get_last_lr()[0]

            if is_main():
                wandb.log(metrics, step=actual_step)
                log.info(f"(epoch {epoch_idx}) (batch {batch_idx}/{len(dataloader)})")
                for k, v in metrics.items():
                    if k.startswith("loss_") or k == "lr":
                        log.info(f"{k}: {v}")

            if actual_step % cfg.train.ckpt_every == 0:
                if is_main():
                    log.info(f"(epoch {epoch_idx}) Saving latest ckpt at step {actual_step}")
                    if os.path.exists(cfg.path_ckpt_latest):
                        try:
                            os.replace(cfg.path_ckpt_latest, cfg.path_ckpt_2nd_latest)
                        except Exception as e:
                            log.info(f"Failed to move latest checkpoint to 2nd latest: {e}")
                    checkpoint = {
                        'step': actual_step,
                        'model': inner.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }
                    torch.save(checkpoint, cfg.path_ckpt_latest)
                barrier(world_size)
            actual_step += 1

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
