"""
Drifting loss. This is a Pytorch version of the original JAX drifting loss https://github.com/lambertae/drifting/blob/main/drift_loss.py
that is adapted to treat batch elements independently.
"""
import torch
import math
import logging
log = logging.getLogger(__name__)

def cdist(x, y, eps=1e-8):
    # [B, N, D] x [B, M, D] -> [B, N, M]
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))

def drift_loss(
    gen,
    fixed_pos,
    fixed_neg=None,
    weight_gen=None,
    weight_pos=None,
    weight_neg=None,
    R_list=(0.02, 0.05, 0.2),
    mask_val=1000000.0,
):
    """
    Batch-Independent Drifting Loss.
    This functionally runs mathematically equivalent to processing B elements 
    with a batch_size=1 and concatenating the results. Normalizations are computed 
    per-element preventing batch cross-talk.
    
    Args:
        gen: [B, C_g, S] generated output
        fixed_pos: [B, C_p, S] positive data points to drift towards
        fixed_neg: [B, C_n, S] negative data points to drift away from
        weight_gen: [B, C_g] (optional; if None: weight is 1) importance weights for gen samples
        weight_pos: [B, C_p] (optional; if None: weight is 1)
        weight_neg: [B, C_n] (optional; if None: weight is 1)
        R_list: a list of R values to use for the kernel function
        mask_val: value for masking diagonal
    Returns:
        loss: [B] tensor of loss for each element of the batch
        info: a dict with entries:
            scale: the scale of the loss (float)
            loss_R: the loss for each R value (float)
            min and max of drifting field (float)
    """

    B, C_g, S = gen.shape
    
    if fixed_neg is None:
        fixed_neg = torch.zeros_like(gen[:, :0, :])
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])
        
    gen = gen.to(torch.float32)
    fixed_pos = fixed_pos.to(torch.float32)
    fixed_neg = fixed_neg.to(torch.float32)
    weight_gen = weight_gen.to(torch.float32)
    weight_pos = weight_pos.to(torch.float32)
    weight_neg = weight_neg.to(torch.float32)
    
    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1) 
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    def calculate_scaled_goal_and_factor_indep(old_gen_in, targets_in, targets_w_in):
        """
        Args:
            old_gen_in: [B, C_g, S]
                Generated data points with stopgrad applied (=old_gen)
            targets_in: [B, C_g + C_n + C_p, S]
                Concatenated generated (with stopgrad), negative, and positive data points (=targets)
            targets_w_in: [B, C_g + C_n + C_p]
                Weights for gen, neg, pos data points (=targets_w)
        Returns:
            goal_scaled: [B, C_g, S]
                The target in eq (6), i.e. the loss will be E[||pred - stopgrad(target)||^2]
            scale_inputs: [B, 1, 1]
                Scaling factor S_j to normalize input features (eq 18): tilde(phi_j) = phi_j/S_j
            info:
                Dictionary, containing info["scale"] and info[f"loss_{R}"] for each temperature R
        """
        info = {}
        dist = cdist(old_gen_in, targets_in) # [B, C_g, C_g + C_n + C_p]
        weighted_dist = dist * targets_w_in[:, None, :] # [B, C_g, C_g + C_n + C_p]
        
        # NOTE LOCAL MEAN: compute means along the C dimensions, leaving [B] separate
        scale = weighted_dist.mean(dim=(1, 2)) / targets_w_in.mean(dim=1) # [B]
        info["scale"] = scale
        
        # NOTE Reshape to [B, 1, 1] to broadcast correctly over generated points and feature dimensions
        scale_view = scale.reshape(B, 1, 1)
        scale_inputs = torch.clamp(scale_view / math.sqrt(S), min=1e-3) # [B, 1, 1]
        old_gen_scaled = old_gen_in / scale_inputs
        targets_scaled = targets_in / scale_inputs
        
        # Normalize distance for kernel
        dist_normed = dist / torch.clamp(scale_view, min=1e-3)

        # Masking
        dist_normed.diagonal(dim1=1, dim2=2).fill_(mask_val)

        # Force loop
        force_across_R = torch.zeros_like(old_gen_scaled)
        
        for R in R_list:
            logits = -dist_normed / R

            affinity = torch.nn.functional.softmax(logits, dim=-1)
            aff_transpose = torch.nn.functional.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(affinity * aff_transpose, min=1e-6))
                # shape [B, C_g, C_g+C_n+C_p]
                # affinity[0,r,c] = e^(-dist_normed[0,r,c]/R) / sqrt((row sum) * (col sum))
                # where row sum = sum_{j=0}^{C_g+C_n+C_p-1} e^(-dist_normed[0,r,j]/R)
                #       col sum = sum_{i=0}^{C_g-1} e^(-dist_normed[0,i,c]/R)
            affinity = affinity * targets_w_in[:, None, :]

            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx] # [B, C_g, C_g+C_n]
            aff_pos = affinity[:, :, split_idx:] # [B, C_g, C_p]
            
            sum_pos = torch.sum(aff_pos, dim=-1, keepdim=True) # [B, C_g, 1]
            r_coeff_neg = -aff_neg * sum_pos # [B, C_g, C_g+C_n], coeff of y-
            sum_neg = torch.sum(aff_neg, dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg # [B, C_g, C_p], coeff of y+
            
            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2) # [B, C_g, C_g+C_n+C_p]

            # total_force_R[b,i,x] = R_coeff[b,i,y] * targets_scaled[b,y,x]
            # total_force_R: i.e. like matrix mult [B,C_g,C_g+C_n+C_p] @ [B,C_g+C_n+C_p,S] -> [B, C_g, S]
            # total_force_R[b,i] = the vector equal to sum of W*y for the ith generated point in the bth batch
            total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)
            
            # the drifting field = sum of W*y - sum of W*x
            total_coeffs = R_coeff.sum(dim=-1) # [B,C_g], where total_coeffs[b,i] = sum of W applied to the ith generated pt in the bth batch
            total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled # sum of W*y - sum of W*x 
            
            # NOTE LOCAL MEAN: Normalize force per batch element using dim=(1,2)
            f_norm_val = (total_force_R ** 2).mean(dim=(1, 2)) # [B]
            info[f"loss_{R}"] = f_norm_val
            
            # Calculate min/max per batch element to match batch_size=1 outputs
            info[f"total_force_{R}_min"] = total_force_R.reshape(B, -1).min(dim=1)[0] # [B]
            info[f"total_force_{R}_max"] = total_force_R.reshape(B, -1).max(dim=1)[0] # [B]

            # Drift normalization
            # Goal: normalize force of each temperature, so that the large magnitude forces don't completely dominate the small ones
            force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8)).reshape(B, 1, 1) # [B, 1, 1]
                # force_scale = lambda_j = sqrt(E[1/C_j * ||V_j||^2])
                # the 1/C_j is already there because of the mean operation in f_norm_val = (total_force_R ** 2).mean()
            force_across_R = force_across_R + total_force_R / force_scale
        
        info["force_across_R_min"] = force_across_R.reshape(B, -1).min(dim=1)[0] # [B]
        info["force_across_R_max"] = force_across_R.reshape(B, -1).max(dim=1)[0] # [B]
        goal_scaled = old_gen_scaled + force_across_R
        return goal_scaled, scale_inputs, info

    with torch.no_grad():
        goal_scaled, scale_inputs, info = calculate_scaled_goal_and_factor_indep(old_gen, targets, targets_w)
    
    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled # [B, C_g, S]
    loss = torch.mean(diff ** 2, dim=(-1, -2)) # [B]
    
    for k in info:
        info[k] = info[k].mean().item()
        
    return loss, info