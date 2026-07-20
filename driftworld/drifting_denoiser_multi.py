"""
DriftWorld: Drifting denoiser, which contains the forward pass, drifting loss, and sampling code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from collections import defaultdict
import copy
import math

from data import Batch
from unet_multi.unet_configs import UNet_model_dict
from drifting.drift_loss_indep import drift_loss

log = logging.getLogger(__name__)

class Denoiser(nn.Module):
    """
    Denoiser based on drifting loss.
    It conditions on num_history_frames current+history frames and generates num_future_frames future frames.
    """
    def __init__(self, unet_name: str, # name of U-Net
                 temp_list: tuple, # temperatures for drifting loss
                 n_neg: int, # number of negative samples for drifting loss
                 num_future_frames: int, # number of future frames to predict
                 num_history_frames: int, # number current+history frames to condition on
                 decay: float, # EMA decay
                 ) -> None:
        super().__init__()
        self.inner_model = UNet_model_dict[unet_name](num_history=num_history_frames) # U-Net
        self.temp_list = temp_list
        self.n_neg = n_neg
        self.num_future_frames = num_future_frames
        self.num_history_frames = num_history_frames
        self.decay = decay
        self.ema_model = copy.deepcopy(self.inner_model)
        self.ema_model.requires_grad_(False)

    def drifting_loss(self, gen: torch.Tensor, pos: torch.Tensor):
        """
        Drifting loss: MSE(gen, stopgrad(gen + V)).
        Args:
            gen: Generated samples [B, N, C, H, W]
            pos: Data samples [B, 1, C, H, W]
        Returns:
            scalar drifting loss and dictionary of metrics
        """
        B, N, C, H, W = gen.shape
        gen_flat = gen.reshape(B, N, -1) # [B, N, D], where D=C*H*W
        pos_flat = pos.reshape(B, 1, -1) # [B, 1, D]

        loss, info = drift_loss(gen=gen_flat, fixed_pos=pos_flat, R_list=self.temp_list)
        return loss.mean(), info

    def forward(self, batch: Batch, device):
        """
        Forward pass to train DriftWorld
        Args:
            batch: a batch of data containing
                - 'image': sequence of visual observations o_0, ..., o_{time_steps-1}
                           shape (b, t, c, h, w)
                - 'action': sequence of actions taken at each timestep a_0, ..., a_{time_steps-1}
                           shape (b, t, 2)
            device: device
        Returns:
            scalar loss and dictionary of metrics
        """
        obs = batch['image']
        act = batch['action'].to(device)

        b, t, c, h, w = obs.size()
        n = self.num_future_frames # num of future frames output by one forward pass
        k = self.num_history_frames # num of current+history frames used as context
        cur_idx = k - 1 # index of the current frame o_t within the window
        assert t == n + k

        target_x = obs[:, cur_idx+1:].permute(0, 2, 1, 3, 4).to(device) # (b, n, c, h, w) -> (b, c, n, h, w)
        history = obs[:, 0 : cur_idx+1].permute(0, 2, 1, 3, 4).to(device) # (b, k, c, h, w) -> (b, c, k, h, w)
        actions = act[:, cur_idx : cur_idx+n] # (b, n, 2)

        history = history.repeat_interleave(self.n_neg, dim=0)
        actions = actions.repeat_interleave(self.n_neg, dim=0)

        # j = self.n_neg
        # gen: (j*b, c, n, h, w) generated samples, i.e. "negative" samples for drifting
        noise = torch.randn((self.n_neg * b, c, n, h, w), device=device)
        gen = self.inner_model(noise, history, actions)
            # raw output from the U-Net. the inputs are
            # noise: (j*b, c, n, h, w) noise for future states s_(T+1), ..., s_(T+n)
            # history: (j*b, c, k, h, w) current+history states s_(T-k+1), ..., s_T
            # actions: (j*b, n, 2) actions a_T, ..., a_(T+n-1)

        # compute drifting loss for every timestep separately
        loss = 0
        metrics = defaultdict(float)
        for i in range(n):
            target_x_slice = target_x[:, :, i].reshape((b, 1, c, h, w)) # (b, c, h, w) -> (b, 1, c, h, w)
            gen_slice = gen[:, :, i].reshape((b, self.n_neg, c, h, w)) # (j*b, c, h, w) -> (b, j, c, h, w)
            loss_i, info_i = self.drifting_loss(gen_slice, target_x_slice)
            loss += loss_i
            for key, value in info_i.items():
                metrics[key] += value

        loss /= n
        averages = {key: total / n for key, total in metrics.items()}
        averages['loss_backprop'] = loss.item()
        return loss, averages

    @torch.no_grad()
    def sample(self, history, actions):
        """
        Sample from DriftWorld (EMA weights, random-noise init).
        Args:
            history: (B, C, K, H, W) tensor of the current+history states o_(t-K+1), ..., o_t
            actions: (B, F, 2) tensor of actions a_t, ..., a_(t+F-1)
        Returns:
            (B, F, C, H, W) tensor of predicted future states o_(t+1), ..., o_(t+F)
        """
        B, C, K, H, W = history.size()
        F = actions.shape[1]
        init_tensor = torch.randn((B, C, F, H, W), device=history.device)
        return self.ema_model(init_tensor, history, actions).permute(0, 2, 1, 3, 4)

    @torch.no_grad()
    def sample_autoregressive(self, cur_state, actions):
        """
        Sample autoregressively from DriftWorld (EMA weights, random-noise init).
        Args:
            cur_state: (B, K, C, H, W) history frames o_(t-(K-1)), ..., o_t
            actions: (B, T, 2) actions a_(t-(K-1)), ..., a_(t+T-K).
                     Only the future actions a_t, ..., a_(t+F-1) are used
                     (F = T - (K-1)); the K-1 history actions are ignored.
        Returns:
            (B, T+1, C, H, W) = (B, K+F, C, H, W) tensor:
                - the K init history frames o_(t-(K-1)), ..., o_t
                - followed by the F predicted future frames o_(t+1), ..., o_(t+F)
        """
        B, K, C, H, W = cur_state.size()
        assert K == self.num_history_frames
        cur_idx = K - 1 # index of the current frame o_t within cur_state
        n = self.num_future_frames
        F = actions.shape[1] - cur_idx # number of future frames to predict
        num_iter = math.ceil(F / n)
        log.info(f"Number future frames: F = {F}")
        log.info(f"Number of iterations: num_iter = {num_iter}")

        # autoregressive rollout
        #   i=0: history frames 0:cur_idx+1 => predict cur_idx+1, ..., cur_idx+n
        #   i=1: history frames n:n+cur_idx+1 => predict cur_idx+n+1, ...
        out = torch.zeros((B, cur_idx + 1 + F, C, H, W), device=cur_state.device)
        out[:, :cur_idx+1] = cur_state
        for i in range(num_iter):
            log.info(f"(iter {i}/{num_iter})")
            history_i = out[:, i*n : i*n + cur_idx + 1].permute(0, 2, 1, 3, 4) # (B, C, K, H, W)
            act_i = actions[:, cur_idx + i*n : cur_idx + (i+1)*n] # (B, n, 2)

            gen = self.sample(history_i, act_i) # (B, n, C, H, W)

            out[:, cur_idx + 1 + i*n : cur_idx + 1 + i*n + gen.shape[1]] = gen
        return out

    @torch.no_grad()
    def update_ema(self):
        """
        Updates the EMA parameters.
        ema_new = decay * ema_old + (1 - decay) * current
        """
        for p_ema, p_net in zip(self.ema_model.parameters(), self.inner_model.parameters()):
            p_ema.lerp_(p_net, 1 - self.decay)
