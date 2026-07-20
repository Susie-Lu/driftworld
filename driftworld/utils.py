"""
Utils from https://github.com/han20192019/gpc_code/blob/main/world_model_train_phase_one/utils.py
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import zarr
import os
import logging


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = logging.getLogger(__name__)


def split_batch_by_id(batch, unique_ids):
    split_batches = []

    for unique_id in unique_ids:
        indices = torch.where(batch['id'] == unique_id)[0]
        mini_batch = {
            'image': batch['image'][indices],
            'agent_pos': batch['agent_pos'][indices],
            'action': batch['action'][indices],
            'id': batch['id'][indices]
        }
        split_batches.append(mini_batch)

    return split_batches

def save(nets, models_save_dir):
    if not os.path.exists(models_save_dir):
        os.makedirs(models_save_dir)

    for model_name, model in nets.items():
        model_path = os.path.join(models_save_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"{model_name}.pth saved")

    print("All models have been saved successfully.")




def list_zarr_paths(dataset_path_dir):
    """
    Return full paths to every '.zarr' entry under dataset_path_dir.
    dataset_path_dir may be a single directory (str) or a list/sequence of
    directories. For a list, all zarrs from the first dir come first (sorted),
    then all from the second, etc. Single-string input reproduces the original
    `sorted(os.listdir(...))` behavior exactly.
    """
    if isinstance(dataset_path_dir, (str, bytes, os.PathLike)):
        dirs = [dataset_path_dir]
    else:
        dirs = list(dataset_path_dir)   # ListConfig / list / tuple

    paths = []
    for d in dirs:
        for entry in sorted(os.listdir(d)):
            if entry[-5:] != '.zarr':
                continue
            paths.append(os.path.join(d, entry))
    return paths


def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    """
    Generates indices for sampling fixed-length sequences from a buffer of multiple episodes.
    Args:
        episode_ends: 1D array of integers where each value is the exclusive end index of an episode in the global buffer
        sequence_length: desired length of each sampled window
        pad_before: number of steps to allow the window to start before the actual episode data begins
                    (results in padding by repeating the first frame)
        pad_after: number of steps to allow the window to extend past the actual episode data ends
                    (results in padding by repeating the last frame)
    Returns:
        np.ndarray: (N, 4) array where each row is [buffer_start, buffer_end, sample_start, sample_end]
            - buffer_start/end: Slice indices for the source data buffer
            - sample_start/end: Slice indices for the target sequence array (of size sequence_length) where the data should be placed.
        i.e. we take the slice data_buffer[buffer_start:buffer_end] of data, and put it into a sequence of length sequence_length.
        The indices sample_start/end specify the portion of the sequence that the data fills, i.e. the rest of the sequence consists of padded 0s.
    """
    indices = list()
    for i in range(len(episode_ends)):
        # ith episode spans the interval [episode_ends[i-1], episode_ends[i])
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # the starting index `idx` ranges from min_start to max_start (inclusive)
        # idx = min_start: [-pad_before, -pad_before + sequence_length)
        # idx = max_start: [episode_length - sequence_length + pad_after, episode_length + pad_after)
        # ==> exactly pad_before steps added before the real data, and exactly pad_after steps added after the real data
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            # start_offset:
            #    = 0 if idx>=0
            #    = -idx else
            # end_offset:
            #    = 0 if idx<=episode_length-sequence_length
            #    = idx+sequence_length-episode_length else

            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            # sample_start_idx:
            #    = 0 if idx>=0
            #    = -idx else     This ensures that we start later in the sequence, leaving the first -idx elements equal to 0.
            # sample_end_idx:
            #    = sequence_length idx<=episode_length-sequence_length
            #    = episode_length - idx else     This ensures that we end earlier in the sequence, so that we fill in the episode_length - idx real data and leave the rest equal to 0.
            
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# dataset
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 id:int,
                 num_demos: int,
                 resize_scale: int, 
                 stats=None):

        # read from zarr dataset
        dataset_root = zarr.open(store=dataset_path, mode='r')

        # limit number of demos
        num_max_demos = dataset_root['meta']['episode_ends'][:].shape[0]
        log.info(f"TrainDataset: num demos in dataset = {num_max_demos}")
        if num_demos < num_max_demos:
            num_max_demos = num_demos
        num_max_frames = dataset_root['meta']['episode_ends'][num_max_demos-1]
        log.info(f"TrainDataset: max num of frames in a demo (among the first {num_max_demos}) = {num_max_frames}")

        # float32, [0,255], (N,96,96,3)
        # DO NOT /255 here because PIL Image class needs raw RGB values
        train_image_data = dataset_root['data']['img']
        """
        Original: train_image_data = dataset_root['data']['img'][:num_max_frames]
        When you apply [:num_max_frames] to a Zarr array, Zarr immediately reads all of those chunks from disk, decodes them,
        and returns a gigantic standard NumPy array. By dropping the slice and just passing dataset_root['data']['img'],
        you are instead passing a reference to the Zarr array. Your RAM consumption drops from tens/hundreds of gigabytes
        down to just a few megabytes because the image data remains safely on the disk.

        In __getitem__ method, we pass self.normalized_train_data (which now holds the Zarr array reference) into the sample_sequence function.
        When sample_sequence executes this slice, Zarr calculates exactly which chunks on disk correspond to [buffer_start_idx:buffer_end_idx], loads only those specific frames from disk, and returns them as a standard NumPy array.
        So removing [:num_max_frames] here does not affect any downstream code.
        """

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'agent_pos': dataset_root['data']['state'][:num_max_frames,:2],
            'action': dataset_root['data']['action'][:num_max_frames]
        }
        episode_ends = dataset_root['meta']['episode_ends'][:num_max_demos]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        if stats == None:
            stats = dict()
            normalized_train_data = dict()
            for key, data in train_data.items():
                stats[key] = get_data_stats(data)
                normalized_train_data[key] = normalize_data(data, stats[key])
        else:
            # compute statistics and normalized data to [-1,1]
            normalized_train_data = dict()
            for key, data in train_data.items():
                normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data['image'] = train_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.dataset_path = dataset_path
        self.id = id
        self.resize_scale = resize_scale

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(self.resize_scale),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        The returned item is `batch`, where
            batch['image']: shape (B, pred_horizon, 3, resize_scale, resize_scale)
            batch['agent_pos']: shape (B, pred_horizon, 2)
            batch['action']: shape (B, pred_horizon, 2)
            batch['id']: shape (B,)
        obs_horizon and action_horizon don't change the output shape;
        they only dictate how far the slicing window is allowed to overhang
        the start and end of an episode to apply padding.
        """
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )
    
        images = nsample['image']

        # PIL Image class will convert the range from [0, 255] to [0, 1] by default (tested)
        images = [np.expand_dims(self.transform(image).numpy(), axis=0) for image in images]
        
        # float32, (2,3,resize_scale,resize_scale)
        # with v2.Normalize: range[-1.7, 2.7], otherwise: range[0, 1] 
        images = np.concatenate(images, axis=0)

        # discard unused observations
        nsample['image'] = torch.from_numpy(images)
        nsample['agent_pos'] = torch.from_numpy(nsample['agent_pos'])
        if 'action' in nsample:
            nsample['action'] = torch.from_numpy(nsample['action'])
        nsample["id"] = self.id

        return nsample


class FullVideoDataset(torch.utils.data.Dataset):
    """
    Yields ONE full-length episode per item (variable length), for full-video evaluation.

    Kept entirely separate from TrainDataset so that the existing windowed dataloader behavior
    is byte-for-byte unchanged. Reuses the module-level normalize_data helper and the same image
    transform as TrainDataset.
    """
    def __init__(self,
                 dataset_path: str,
                 id: int,
                 num_demos: int,
                 resize_scale: int,
                 stats):
        dataset_root = zarr.open(store=dataset_path, mode='r')

        episode_ends = dataset_root['meta']['episode_ends'][:]
        num_max_demos = min(num_demos, episode_ends.shape[0])
        episode_ends = episode_ends[:num_max_demos]
        num_max_frames = int(episode_ends[-1])
        log.info(f"FullVideoDataset: using first {num_max_demos} demos ({num_max_frames} frames)")

        # keep the zarr array reference (lazy read), matching TrainDataset's memory strategy
        self.image_data = dataset_root['data']['img']

        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'agent_pos': dataset_root['data']['state'][:num_max_frames, :2],
            'action': dataset_root['data']['action'][:num_max_frames],
        }
        # normalize agent_pos / action to [-1, 1] using the provided stats
        self.normalized = {k: normalize_data(v, stats[k]) for k, v in train_data.items()}

        # episode i spans the half-open interval [start, end)
        self.episodes = []
        for i in range(len(episode_ends)):
            start = 0 if i == 0 else int(episode_ends[i - 1])
            self.episodes.append((start, int(episode_ends[i])))

        self.id = id
        self.resize_scale = resize_scale
        self.transform = v2.Compose([  # identical to TrainDataset
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(self.resize_scale),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        """
        Returns a single full-length episode:
            'image':     (T, 3, resize_scale, resize_scale) float32 in [0, 1]
            'agent_pos': (T, 2) float32 in [-1, 1]
            'action':    (T, 2) float32 in [-1, 1]
            'id':        int
            'length':    T (episode length)
        where T is the episode's natural length (variable across episodes).
        """
        start, end = self.episodes[idx]

        imgs = self.image_data[start:end]  # (T, 96, 96, 3) uint8 numpy in [0, 255]
        imgs = np.concatenate(
            [np.expand_dims(self.transform(image).numpy(), axis=0) for image in imgs], axis=0)

        return {
            'image': torch.from_numpy(imgs),
            'agent_pos': torch.from_numpy(self.normalized['agent_pos'][start:end]),
            'action': torch.from_numpy(self.normalized['action'][start:end]),
            'id': self.id,
            'length': end - start,
        }




class StateDictMixin:
    def _init_fields(self) -> None:
        def has_sd(x: str) -> bool:
            return callable(getattr(x, "state_dict", None)) and callable(getattr(x, "load_state_dict", None))

        self._all_fields = {k for k in vars(self) if not k.startswith("_")}
        self._fields_sd = {k for k in self._all_fields if has_sd(getattr(self, k))}

    def _get_field(self, k: str) -> Any:
        return getattr(self, k).state_dict() if k in self._fields_sd else getattr(self, k)

    def _set_field(self, k: str, v: Any) -> None:
        getattr(self, k).load_state_dict(v) if k in self._fields_sd else setattr(self, k, v)

    def state_dict(self) -> Dict[str, Any]:
        if not hasattr(self, "_all_fields"):
            self._init_fields()
        return {k: self._get_field(k) for k in self._all_fields}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not hasattr(self, "_all_fields"):
            self._init_fields()
        assert set(list(state_dict.keys())) == self._all_fields
        for k, v in state_dict.items():
            self._set_field(k, v)

from torch import Tensor
ComputeLossOutput = Tuple[Tensor, Dict[str, Any]]