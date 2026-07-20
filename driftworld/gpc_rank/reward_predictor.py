"""
Reward Predictor, based on https://github.com/han20192019/gpc_code/blob/main/gpc_rank_evaluation/eval_baseline.py
"""

import torch
import torch.nn as nn
import torchvision.models as models

class RewardPredictor(nn.Module):
    """
    Model based on ResNet18 that is used to estimate the spatial state of the T-block from the input image.
    
    Usage:
        - predict (x, y) coordinates
        - predict (cos(theta), sin(theta)) orientation of the block
    """
    def __init__(self):
        super(RewardPredictor, self).__init__()
        # Load the pretrained ResNet18 model
        self.resnet18 = models.resnet18(weights=None)
        
        # Remove the final fully connected layer (classifier)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        
        # Define the MLP for (x, y) or (cos(theta), sin(theta)) regression
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) image tensor
        Returns:
            pose: (B, 2) predicted 2D tensor of the (x,y) or (cos(theta), sin(theta)) pose
        """
        features = self.resnet18(x)
        pose = self.mlp(features)
        return pose

def transform_vertices_torch(px: torch.Tensor, py: torch.Tensor, ptheta: torch.Tensor, 
                           vertices1: torch.Tensor, vertices2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a 2D rigid body transformation (rotation followed by translation) to two sets of vertices.
    Inputs:
        px (torch.Tensor): The x-coordinate translation (scalar).
        py (torch.Tensor): The y-coordinate translation (scalar).
        ptheta (torch.Tensor): The rotation angle in radians (scalar).
        vertices1 (torch.Tensor): The first set of vertices, shape (N, 2).
        vertices2 (torch.Tensor): The second set of vertices, shape (M, 2).
    Outputs:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed vertices1 and vertices2, 
        with shapes (N, 2) and (M, 2) respectively.
    """
    rotation_matrix = torch.vstack([
        torch.stack((torch.cos(ptheta), -torch.sin(ptheta))),
        torch.stack((torch.sin(ptheta), torch.cos(ptheta)))
    ])

    # Perform rotation
    new_vertices1 = vertices1 @ rotation_matrix
    new_vertices2 = vertices2 @ rotation_matrix
    
    # Perform translation
    translation = torch.stack([px, py])
    new_vertices1 = new_vertices1 + translation
    new_vertices2 = new_vertices2 + translation
    
    return new_vertices1, new_vertices2

def estimate_reward_torch(block_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    """
    Calculates a distance-based penalty (reward) between the current block pose and the target goal pose.
    It computes the sum of Euclidean distances between the corresponding vertices of the T-block at both poses.
    Inputs:
        block_pose: (3,) tensor of the current [x, y, theta] pose of the block
        target_pose: (3,) tensor of the target [x, y, theta] pose of the block
    Outputs:
        reward: a scalar tensor representing the computed reward (penalty) value
    """
    # Convert initial vertices to torch tensors
    vertices1 = torch.tensor([[-10.0, 2.5], [10.0, 2.5], [10.0, 7.5], [-10.0, 7.5]], dtype=torch.float32, device=block_pose.device) * 6.0
    vertices2 = torch.tensor([[-2.5, 2.5], [-2.5, -12.5], [2.5, -12.5], [2.5, 2.5]], dtype=torch.float32, device=block_pose.device) * 6.0

    # Transform vertices for both block and goal
    block_verts1, block_verts2 = transform_vertices_torch(
        block_pose[0], block_pose[1], block_pose[2], 
        vertices1, vertices2
    )
    goal_verts1, goal_verts2 = transform_vertices_torch(
        target_pose[0], target_pose[1], target_pose[2], 
        vertices1, vertices2
    )

    # Concatenate vertices
    block_verts = torch.cat([block_verts1, block_verts2], dim=0)
    goal_verts = torch.cat([goal_verts1, goal_verts2], dim=0)

    # Calculate distances and reward
    dist_sum = torch.norm(block_verts - goal_verts, dim=1).sum()
    reward = 0.01 * dist_sum

    return reward
