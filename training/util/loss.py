# @ GonzaloMartinGarcia
# Task specific loss functions are from Depth Anything https://github.com/LiheYoung/Depth-Anything.
# Modifications have been made to improve numerical stability for this project (marked by '# add').

import torch
import torch.nn as nn
import torch.nn.functional as F

#########
# Losses
#########

# Scale and Shift Invariant Loss
class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"
    def forward(self, prediction, target, mask):
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        prediction, target = prediction.squeeze(1), target.squeeze(1)
        # add
        with torch.autocast(device_type='cuda', enabled=False):
            prediction = prediction.float()
            target     = target.float()

            scale, shift = compute_scale_and_shift_masked(prediction, target, mask)
            scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
            loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
        return loss
    
def compute_scale_and_shift_masked(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)
    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0 #1e-3
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    return x_0, x_1


# Angluar Loss
class AngularLoss(nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()
        self.name = "Angular"

    def forward(self, prediction, target, mask=None):
        with torch.autocast(device_type='cuda', enabled=False):
            prediction = prediction.float()
            target = target.float()
            mask   = mask[:,0,:,:]    
            dot_product = torch.sum(prediction * target, dim=1)
            dot_product = torch.clamp(dot_product, -1.0, 1.0)
            angle = torch.acos(dot_product)
            if mask is not None:
                angle = angle[mask]
            loss = angle.mean()
        return loss

# Texture Loss
class TextureLoss(nn.Module):
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1):
        super(TextureLoss, self).__init__()
        self.name = "TextureLoss"
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
    def forward(self, prediction, target, mask=None):
        """
        Compute loss between predicted texture and target texture
        
        Args:
            prediction: Predicted texture map [B, 3, H, W]
            target: Target texture map [B, 3, H, W]
            mask: Optional mask for valid pixels [B, 1, H, W]
            
        Returns:
            Combined loss value
        """
        with torch.autocast(device_type='cuda', enabled=False):
            prediction = prediction.float()
            target = target.float()
            
            # Apply mask if provided
            if mask is not None:
                if mask.ndim == 4:
                    # Expand mask to match RGB channels
                    mask_expanded = mask.expand(-1, 3, -1, -1)
                    
                    # Check if mask has any valid pixels
                    if mask_expanded.any():
                        # L1 loss on masked pixels
                        l1_loss = F.l1_loss(
                            prediction[mask_expanded], 
                            target[mask_expanded]
                        )
                    else:
                        l1_loss = torch.tensor(0.0, device=prediction.device)
                else:
                    # Handle case where mask is not 4D
                    l1_loss = F.l1_loss(prediction, target)
            else:
                # No mask, compute loss on all pixels
                l1_loss = F.l1_loss(prediction, target)
            
            # Simple perceptual loss using differences at multiple scales
            perceptual_loss = torch.tensor(0.0, device=prediction.device)
            if self.perceptual_weight > 0:
                for scale in [1, 2, 4]:
                    if scale > 1:
                        # Downsample both prediction and target
                        pred_down = F.avg_pool2d(prediction, kernel_size=scale, stride=scale)
                        target_down = F.avg_pool2d(target, kernel_size=scale, stride=scale)
                        
                        # Compute L1 loss at this scale
                        scale_loss = F.l1_loss(pred_down, target_down)
                        perceptual_loss += scale_loss
                
                # Average the perceptual loss across scales
                perceptual_loss /= 3.0
            
            # Combine losses
            total_loss = self.l1_weight * l1_loss + self.perceptual_weight * perceptual_loss
            
        return total_loss
