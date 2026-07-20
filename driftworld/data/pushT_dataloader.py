import torch
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from utils import TrainDataset, FullVideoDataset, list_zarr_paths


def get_pushT_loader(cfg, rank=0, world_size=1):
    """
    Returns dataloader for Push-T dataset.
    When world_size > 1, a DistributedSampler shards the dataset across ranks;
    otherwise plain shuffling is used. batch_size is the PER-GPU batch size.
    """
    dataset_list = []
    combined_stats = []
    num_datasets = 0
    all_data_stats = {'agent_pos': {'min': np.array([2.0407837e-04, 1.0189312e+00], dtype=np.float32), 'max': np.array([509.08173, 509.43417], dtype=np.float32)}, 'action': {'min': np.array([0., 0.], dtype=np.float32), 'max': np.array([511., 511.], dtype=np.float32)}}

    for full_path in list_zarr_paths(cfg.data.dataset_path_dir):
        # create dataset from file
        dataset = TrainDataset(
            dataset_path=full_path,
            pred_horizon=cfg.data.pred_horizon,
            obs_horizon=cfg.data.obs_horizon,
            action_horizon=cfg.data.action_horizon,
            id = num_datasets,
            num_demos = cfg.data.num_train_demos,
            resize_scale = cfg.data.resize_scale,
            stats = all_data_stats
        )
        num_datasets += 1
        # save training data statistics (min, max) for each dim
        stats = dataset.stats
        dataset_list.append(dataset)
        combined_stats.append(stats)

    combined_dataset = ConcatDataset(dataset_list)

    if world_size > 1:
        sampler = DistributedSampler(
            combined_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        dataloader = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.dataloader.num_workers,
            prefetch_factor=cfg.dataloader.prefetch_factor,
            sampler=sampler,
            pin_memory=cfg.dataloader.pin_memory,
            persistent_workers=True
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.dataloader.num_workers,
            prefetch_factor=cfg.dataloader.prefetch_factor,
            shuffle=True,
            pin_memory=cfg.dataloader.pin_memory,
            persistent_workers=True
        )
    return dataloader

def get_pushT_loader_shuffleFalse(cfg):
    """
    Returns dataloader for Push-T dataset with shuffle=False.
    """
    dataset_list = []
    num_datasets = 0
    all_data_stats = {'agent_pos': {'min': np.array([2.0407837e-04, 1.0189312e+00], dtype=np.float32), 'max': np.array([509.08173, 509.43417], dtype=np.float32)}, 'action': {'min': np.array([0., 0.], dtype=np.float32), 'max': np.array([511., 511.], dtype=np.float32)}}

    for full_path in list_zarr_paths(cfg.data.dataset_path_dir):
        dataset = TrainDataset(
            dataset_path=full_path,
            pred_horizon=cfg.data.pred_horizon,
            obs_horizon=cfg.data.obs_horizon,
            action_horizon=cfg.data.action_horizon,
            id = num_datasets,
            num_demos = cfg.data.num_train_demos,
            resize_scale = cfg.data.resize_scale,
            stats = all_data_stats
        )
        num_datasets += 1
        dataset_list.append(dataset)

    combined_dataset = ConcatDataset(dataset_list)

    dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        shuffle=False,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=True
    )
    return dataloader

def get_pushT_full_loader(cfg):
    """
    Returns a dataloader that yields full-length episodes for Push-T.
    Uses batch_size=1 and shuffle=False.
    """
    dataset_list = []
    num_datasets = 0
    all_data_stats = {'agent_pos': {'min': np.array([2.0407837e-04, 1.0189312e+00], dtype=np.float32), 'max': np.array([509.08173, 509.43417], dtype=np.float32)}, 'action': {'min': np.array([0., 0.], dtype=np.float32), 'max': np.array([511., 511.], dtype=np.float32)}}

    for full_path in list_zarr_paths(cfg.data.dataset_path_dir):
        dataset = FullVideoDataset(
            dataset_path=full_path,
            id=num_datasets,
            num_demos=cfg.data.num_train_demos,
            resize_scale=cfg.data.resize_scale,
            stats=all_data_stats
        )
        num_datasets += 1
        dataset_list.append(dataset)

    combined_dataset = ConcatDataset(dataset_list)

    dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=1,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        shuffle=False,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=True
    )
    return dataloader
