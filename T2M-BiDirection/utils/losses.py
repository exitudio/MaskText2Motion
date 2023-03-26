import torch
import torch.nn as nn

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints, vqvae_transformer=False):
        super(ReConsLoss, self).__init__()
        
        reduction='sum' if vqvae_transformer else 'mean'
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss(reduction=reduction)
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss(reduction=reduction)
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss(reduction=reduction)
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4
        
    def forward(self, motion_pred, motion_gt, src_mask=None) : 
        motion_pred = motion_pred[..., : self.motion_dim]
        motion_gt = motion_gt[..., :self.motion_dim]
        if src_mask is not None:
             # [INFO] motion_pred and motion_gt already masked from outside
            loss = self.Loss(motion_pred , motion_gt ) / \
                (src_mask.sum()*motion_gt.shape[-1])
        else:
            loss = self.Loss(motion_pred, motion_gt)
        return loss
    
    def forward_vel(self, motion_pred, motion_gt, src_mask=None) : 
        pred_j_pos = motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4]
        gt_j_pos = motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4]
        if src_mask is not None:
            # [INFO] motion_pred and motion_gt already masked from outside
            loss = self.Loss(pred_j_pos , gt_j_pos ) / \
                (src_mask.sum()*gt_j_pos.shape[-1])
        else:
            loss = self.Loss(pred_j_pos, gt_j_pos)
        return loss
    
    