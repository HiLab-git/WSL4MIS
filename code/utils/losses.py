import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from utils import losses, metrics, ramps,util

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class pDLoss(nn.Module):
    def __init__(self, n_classes, ignore_index):
        super(pDLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore_mask):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target * ignore_mask)
        y_sum = torch.sum(target * target * ignore_mask)
        z_sum = torch.sum(score * score * ignore_mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        ignore_mask = torch.ones_like(target)
        ignore_mask[target == self.ignore_index] = 0
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore_mask)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


class SizeLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(SizeLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        output_counts = torch.sum(torch.softmax(output, dim=1), dim=(2, 3))
        target_counts = torch.zeros_like(output_counts)
        for b in range(0, target.shape[0]):
            elements, counts = torch.unique(
                target[b, :, :, :, :], sorted=True, return_counts=True)
            assert torch.numel(target[b, :, :, :, :]) == torch.sum(counts)
            target_counts[b, :] = counts

        lower_bound = target_counts * (1 - self.margin)
        upper_bound = target_counts * (1 + self.margin)
        too_small = output_counts < lower_bound
        too_big = output_counts > upper_bound
        penalty_small = (output_counts - lower_bound) ** 2
        penalty_big = (output_counts - upper_bound) ** 2
        # do not consider background(i.e. channel 0)
        res = too_small.float()[:, 1:] * penalty_small[:, 1:] + \
            too_big.float()[:, 1:] * penalty_big[:, 1:]
        loss = res / (output.shape[2] * output.shape[3] * output.shape[4])
        return loss.mean()


class MumfordShah_Loss(nn.Module):
    def levelsetLoss(self, output, target, penalty='l1'):
        # input size = batch x 1 (channel) x height x width
        outshape = output.shape
        tarshape = target.shape
        self.penalty = penalty
        loss = 0.0
        for ich in range(tarshape[1]):
            target_ = torch.unsqueeze(target[:, ich], 1)
            target_ = target_.expand(
                tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pcentroid = torch.sum(target_ * output, (2, 3)
                                  ) / torch.sum(output, (2, 3))
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - \
                pcentroid.expand(
                    tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss)
        return loss

    def gradientLoss2d(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if self.penalty == "l2":
            dH = dH * dH
            dW = dW * dW

        loss = torch.sum(dH) + torch.sum(dW)
        return loss

    def forward(self, image, prediction):
        loss_level = self.levelsetLoss(image, prediction)
        loss_tv = self.gradientLoss2d(prediction)
        return loss_level + loss_tv

def scc_loss(cos_sim,tau,lb_center_12_bg,
             lb_center_12_a,un_center_12_bg, 
             un_center_12_a,lb_center_12_b,lb_center_12_c,un_center_12_b,un_center_12_c):
    
    loss_intra_bg = torch.exp((cos_sim(lb_center_12_bg, un_center_12_bg))/tau)
    loss_intra_la = torch.exp((cos_sim(lb_center_12_a, un_center_12_a))/tau)
    loss_intra_lb = torch.exp((cos_sim(lb_center_12_b, un_center_12_b))/tau)
    loss_intra_lc = torch.exp((cos_sim(lb_center_12_c, un_center_12_c))/tau)


    loss_inter_bg_la = torch.exp((cos_sim(lb_center_12_bg, un_center_12_a))/tau)
    loss_inter_bg_lb = torch.exp((cos_sim(lb_center_12_bg, un_center_12_b))/tau)
    loss_inter_bg_lc = torch.exp((cos_sim(lb_center_12_bg, un_center_12_c))/tau)


    loss_inter_la_bg = torch.exp((cos_sim(lb_center_12_a, un_center_12_bg))/tau)
    loss_inter_lb_bg = torch.exp((cos_sim(lb_center_12_b, un_center_12_bg))/tau)
    loss_inter_lc_bg = torch.exp((cos_sim(lb_center_12_c, un_center_12_bg))/tau)


    loss_contrast_bg = -torch.log(loss_intra_bg)+torch.log(loss_inter_bg_la)+torch.log(loss_inter_bg_lb)+torch.log(loss_inter_bg_lc)
    loss_contrast_la = -torch.log(loss_intra_la)+torch.log(loss_inter_la_bg)+torch.log(loss_inter_lb_bg)+torch.log(loss_inter_lc_bg)
    loss_contrast_lb = -torch.log(loss_intra_lb)+torch.log(loss_inter_la_bg)+torch.log(loss_inter_lb_bg)+torch.log(loss_inter_lc_bg)
    loss_contrast_lc = -torch.log(loss_intra_lc)+torch.log(loss_inter_la_bg)+torch.log(loss_inter_lb_bg)+torch.log(loss_inter_lc_bg)
    
    loss_contrast = torch.mean(loss_contrast_bg+loss_contrast_la+loss_contrast_lb+loss_contrast_lc)
    return loss_contrast

def get_aff_loss(inputs, targets):

    # pos_label = (targets == 1).type(torch.int16)
    # pos_label2 = (targets == 2).type(torch.int16)
    # pos_label3 = (targets == 3).type(torch.int16)

    # pos_count = pos_label.sum() + 1
    # pos_count2 = pos_label2.sum() + 1
    # pos_count3 = pos_label3.sum() + 1

    # neg_label = (targets == 0).type(torch.int16)
    # neg_count = neg_label.sum() + 1
    # #inputs = torch.sigmoid(input=inputs)

    # pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    # pos_loss2 = torch.sum(pos_label2 * (1 - inputs)) / pos_count2   
    # pos_loss3 = torch.sum(pos_label3 * (1 - inputs)) / pos_count3 

    # neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    # return 0.5 * (pos_loss+pos_loss2+pos_loss3) + 0.5 * neg_loss
    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1
    #inputs = torch.sigmoid(input=inputs)

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss


class SegformerAffinityEnergyLoss(nn.Module):
    def __init__(self, ):
        super(SegformerAffinityEnergyLoss, self).__init__()


        self.weight = 0.78
        self.class_num = 4
        self.loss_index = 3

        #self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        #self.tree_filter_layers = TreeFilter2D(groups=1, sigma=self.configer.get('tree_loss', 'sigma'))

    # [bz,21,128,128], [bz,21,16,16], [bz,21,32,32], [bz,21,64,64], _attns- a list of 4
    def forward(self, outputs, low_feats, unlabeled_ROIs, targets,ema_att,max_iterations,iter_num): 

       
        seg, seg_16, seg_32, seg_64, = outputs
        attns =low_feats
        bz_label,_,_= targets.size()
            
        bz, _, token_b1_n1, token_b1_n2  = attns[0][0].size()

        attn_avg1 = torch.zeros(bz, token_b1_n1, token_b1_n2, dtype=seg.dtype, device=seg.device)
        for attn in attns[0]:
            attn = attn.mean(dim=1)
            attn = attn / attn.sum(dim=-1,keepdim=True)
            attn_avg1 += attn
        attn_avg1 = attn_avg1 / len(attns[0])
        
        # attn_avg2 [bz, 64*64, 16*16]
        bz, _, token_b2_n1, token_b2_n2 = attns[1][0].size()
        attn_avg2 = torch.zeros(bz, token_b2_n1, token_b2_n2, dtype=seg.dtype, device=seg.device)
        for attn in attns[1]:
            attn = attn.mean(dim=1)
            attn = attn / attn.sum(dim=-1,keepdim=True)
            attn_avg2 += attn
        attn_avg2 = attn_avg2 / len(attns[1])

        # attn_avg3 [bz, 32*32, 16*16]
        bz, _, token_b3_n1, token_b3_n2 = attns[2][0].size()
        attn_avg3 = torch.zeros(bz, token_b3_n1, token_b3_n2, dtype=seg.dtype, device=seg.device)
        for attn in attns[2]:
            attn = attn.mean(dim=1)
            attn = attn / attn.sum(dim=-1,keepdim=True)
            attn_avg3 += attn
        attn_avg3 = attn_avg3 / len(attns[2]) 

        # attn_avg4 [bz, 32*32, 32*32]
        bz, _, token_b4_n1, token_b4_n2 = attns[3][0].size()
        attn_avg4 = torch.zeros(bz, token_b4_n1, token_b4_n2, dtype=seg.dtype, device=seg.device)
        for attn in attns[3]:
            attn = attn.mean(dim=1)
            attn = attn / attn.sum(dim=-1,keepdim=True)
            attn_avg4 += attn
        attn_avg4 = attn_avg4 / len(attns[3])     

        #TODO: uncertrainty
        # bz_ema, _, token_b1_n1, token_b1_n2  = ema_att[0][0].size()

        attn_avg1_ema = torch.zeros(bz, token_b1_n1, token_b1_n2, dtype=seg.dtype, device=seg.device)
        for attn in ema_att[0]:
            attn = attn.mean(dim=1)
            attn = attn / attn.sum(dim=-1,keepdim=True)
            attn_avg1_ema += attn
        attn_avg1_ema = attn_avg1_ema / len(attns[0])
        
        # attn_avg2 [bz, 64*64, 16*16]
        # bz, _, token_b2_n1, token_b2_n2 = attns[1][0].size()
        attn_avg2_ema = torch.zeros(bz, token_b2_n1, token_b2_n2, dtype=seg.dtype, device=seg.device)
        for attn in attns[1]:
            attn = attn.mean(dim=1)
            attn = attn / attn.sum(dim=-1,keepdim=True)
            attn_avg2_ema += attn
        attn_avg2_ema = attn_avg2_ema / len(attns[1])

        # attn_avg3 [bz, 32*32, 16*16]
        # bz, _, token_b3_n1, token_b3_n2 = attns[2][0].size()
        attn_avg3_ema = torch.zeros(bz, token_b3_n1, token_b3_n2, dtype=seg.dtype, device=seg.device)
        for attn in attns[2]:
            attn = attn.mean(dim=1)
            attn = attn / attn.sum(dim=-1,keepdim=True)
            attn_avg3_ema += attn
        attn_avg3_ema = attn_avg3_ema / len(attns[2]) 

        # attn_avg4 [bz, 32*32, 32*32]
        # bz, _, token_b4_n1, token_b4_n2 = attns[3][0].size()
        attn_avg4_ema = torch.zeros(bz, token_b4_n1, token_b4_n2, dtype=seg.dtype, device=seg.device)
        for attn in attns[3]:
            attn = attn.mean(dim=1)
            attn = attn / attn.sum(dim=-1,keepdim=True)
            attn_avg4_ema += attn
        attn_avg4_ema = attn_avg4_ema / len(attns[3])     




        # soft affinity probability 
        _, _, h128,w128    = seg.size()
        prob128            = torch.softmax(seg, dim=1)            # prob-[bz,21,128,128]
        prob128            = prob128.view(bz,self.class_num,-1).permute(0,2,1)  # [bz, 128*128, 21]
        prob128_softmax    = torch.softmax(prob128, dim=-1)

        _, _, h16,w16      = seg_16.size()
        prob16             = torch.softmax(seg_16, dim=1)           
        prob16             = prob16.view(bz,self.class_num,-1).permute(0,2,1)  
        prob16_softmax     = torch.softmax(prob16, dim=-1)

        _, _, h32,w32      = seg_32.size()
        prob32             = torch.softmax(seg_32, dim=1)            
        prob32             = prob32.view(bz,self.class_num,-1).permute(0,2,1)  
        prob32_softmax     = torch.softmax(prob32, dim=-1)

        _, _, h64,w64      = seg_64.size()
        prob64             = torch.softmax(seg_64, dim=1)            
        prob64             = prob64.view(bz,self.class_num,-1).permute(0,2,1)  
        prob64_softmax     = torch.softmax(prob64, dim=-1)



        # attn_avg1_ema= F.interpolate(attn_avg1_ema, size=targets.shape[1:], mode='bilinear', align_corners=False)
        # attn_avg2_ema= F.interpolate(attn_avg2_ema, size=targets.shape[1:], mode='bilinear', align_corners=False)
        # attn_avg3_ema= F.interpolate(attn_avg3_ema, size=targets.shape[1:], mode='bilinear', align_corners=False)
        # attn_avg1_ema= F.interpolate(attn_avg4_ema, size=targets.shape[1:], mode='bilinear', align_corners=False)

        # attn_avg_ema = torch.mean(torch.stack([F.softmax(attn_avg1_ema, dim=1), F.softmax(attn_avg2_ema, dim=1)]), dim=0)

        threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)

        uncertainty1 = -1.0 * torch.sum(attn_avg1_ema.permute(0,2,1) * torch.log(attn_avg1_ema.permute(0,2,1) + 1e-6), dim=1, keepdim=True)
        uncertainty_mask1 = (uncertainty1 > threshold)
        # pusdo_att = torch.argmax(F.softmax(pusdo_label, dim=1).detach(), dim=1, keepdim=True).float()    
        certainty_att1 = attn_avg1.permute(0,2,1).clone()
        uncertainty_mask1 = uncertainty_mask1.repeat(1,64,1)
        certainty_att1[uncertainty_mask1] = 0
        

        uncertainty2 = -1.0 * torch.sum(attn_avg2_ema.permute(0,2,1) * torch.log(attn_avg2_ema.permute(0,2,1) + 1e-6), dim=1, keepdim=True)
        uncertainty_mask2 = (uncertainty2 > threshold)
        # pusdo_att = torch.argmax(F.softmax(pusdo_label, dim=1).detach(), dim=1, keepdim=True).float()    
        certainty_att2 = attn_avg2.permute(0,2,1).clone()
        uncertainty_mask2 = uncertainty_mask2.repeat(1,64,1)
        certainty_att2[uncertainty_mask2] = 0
        
        
        uncertainty3 = -1.0 * torch.sum(attn_avg3_ema.permute(0,2,1) * torch.log(attn_avg3_ema.permute(0,2,1) + 1e-6), dim=1, keepdim=True)
        

        uncertainty_mask3 = (uncertainty3 > threshold)
        # pusdo_att = torch.argmax(F.softmax(pusdo_label, dim=1).detach(), dim=1, keepdim=True).float()    
        certainty_att3 = attn_avg3.permute(0,2,1).clone()
        
        uncertainty_mask3 = uncertainty_mask3.repeat(1,64,1)
        certainty_att3[uncertainty_mask3] = 0


        uncertainty4 = -1.0 * torch.sum(attn_avg4_ema.permute(0,2,1) * torch.log(attn_avg4_ema.permute(0,2,1) + 1e-6), dim=1, keepdim=True)
        uncertainty_mask4 = (uncertainty4 > threshold)
        # pusdo_att = torch.argmax(F.softmax(pusdo_label, dim=1).detach(), dim=1, keepdim=True).float()    
        certainty_att4 = attn_avg4.permute(0,2,1).clone()
        
        uncertainty_mask4 = uncertainty_mask4.repeat(1,256,1)
        certainty_att4[uncertainty_mask4] = 0


        # loss
        # affinity_loss1     = torch.abs(torch.matmul(attn_avg1, prob16) - prob128)  # [bz, 128*128, 21]
        # affinity_loss2     = torch.abs(torch.matmul(attn_avg2, prob16) - prob64)
        # affinity_loss3     = torch.abs(torch.matmul(attn_avg3, prob16) - prob32)
        # affinity_loss4     = torch.abs(torch.matmul(attn_avg4, prob32) - prob32)
        pusdo_label1= torch.softmax(torch.matmul(certainty_att1.permute(0,2,1), prob16),dim=-1)
        pusdo_label1= pusdo_label1.view(bz,self.class_num,h128,w128)
        pusdo_label1= F.interpolate(pusdo_label1, size=targets.shape[1:], mode='bilinear', align_corners=False)

        pusdo_label2=torch.softmax(torch.matmul(certainty_att2.permute(0,2,1), prob16),dim=-1)
        pusdo_label2= pusdo_label2.view(bz,self.class_num,h64,w64)
        pusdo_label2= F.interpolate(pusdo_label2, size=targets.shape[1:], mode='bilinear', align_corners=False)
        # return pusdo_label
        pusdo_label3=torch.softmax(torch.matmul(certainty_att3.permute(0,2,1), prob16),dim=-1)
        pusdo_label3= pusdo_label3.view(bz,self.class_num,h32,w32)
        pusdo_label3= F.interpolate(pusdo_label3, size=targets.shape[1:], mode='bilinear', align_corners=False)

        pusdo_label4=torch.softmax(torch.matmul(certainty_att4.permute(0,2,1), prob32),dim=-1)
        pusdo_label4= pusdo_label4.view(bz,self.class_num,h32,w32)
        pusdo_label4= F.interpolate(pusdo_label4, size=targets.shape[1:], mode='bilinear', align_corners=False)
        pusdo_label=(pusdo_label1+pusdo_label2+pusdo_label3+pusdo_label4)/4

        affinity_loss1     = torch.abs(torch.softmax(torch.matmul(attn_avg1[:bz_label,...], 
                                                                  prob16[:bz_label,...]),dim=-1) - prob128_softmax[:bz_label,...])  # [bz, 128*128, 21]
        affinity_loss2     = torch.abs(torch.softmax(torch.matmul(attn_avg2[:bz_label,...], 
                                                                  prob16[:bz_label,...]),dim=-1) - prob64_softmax[:bz_label,...])
        affinity_loss3     = torch.abs(torch.softmax(torch.matmul(attn_avg3[:bz_label,...], 
                                                                  prob16[:bz_label,...]),dim=-1) - prob32_softmax[:bz_label,...])
        affinity_loss4     = torch.abs(torch.softmax(torch.matmul(attn_avg4[:bz_label,...], 
                                                                  prob32[:bz_label,...]),dim=-1) - prob32_softmax[:bz_label,...])

        # affinity_loss1     = F.kl_div(F.log_softmax(torch.matmul(attn_avg1, prob16),dim=-1) , prob128_softmax)  # [bz, 128*128, 21]
        # affinity_loss2     = F.kl_div(F.log_softmax(torch.matmul(attn_avg2, prob16),dim=-1) , prob64_softmax)
        # affinity_loss3     = F.kl_div(F.log_softmax(torch.matmul(attn_avg3, prob16),dim=-1) , prob32_softmax)
        # affinity_loss4     = F.kl_div(F.log_softmax(torch.matmul(attn_avg4, prob32),dim=-1) , prob32_softmax)

        # # affinity loss number
        with torch.no_grad():
            unlabeled_ROIs128 = F.interpolate(unlabeled_ROIs.unsqueeze(1), size=(h128, w128), mode='nearest')  # [bz, 1, 128, 128]
            unlabeled_ROIs128 = unlabeled_ROIs128.view(bz_label, -1).unsqueeze(-1)
            N128 = unlabeled_ROIs128.sum()

            unlabeled_ROIs16 = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h16, w16), mode='nearest')  # [bz, 1, 16, 16]
            unlabeled_ROIs16 = unlabeled_ROIs16.view(bz, -1).unsqueeze(-1)
            N16 = unlabeled_ROIs16.sum()

            unlabeled_ROIs32 = F.interpolate(unlabeled_ROIs.unsqueeze(1), size=(h32, w32), mode='nearest')  # [bz, 1, 16, 16]
            unlabeled_ROIs32 = unlabeled_ROIs32.view(bz_label, -1).unsqueeze(-1)
            N32 = unlabeled_ROIs32.sum()

            unlabeled_ROIs64 = F.interpolate(unlabeled_ROIs.unsqueeze(1), size=(h64, w64), mode='nearest')  # [bz, 1, 16, 16]
            unlabeled_ROIs64 = unlabeled_ROIs64.view(bz_label, -1).unsqueeze(-1)
            N64 = unlabeled_ROIs64.sum()

        if N128>0:
            affinity_loss1 = (unlabeled_ROIs128 * affinity_loss1).sum() / N128
        if N64>0:
            affinity_loss2 = (unlabeled_ROIs64 * affinity_loss2).sum() / N64
        if N32>0:
            affinity_loss3 = (unlabeled_ROIs32 * affinity_loss3).sum() / N32
        if N32>0:
            affinity_loss4 = (unlabeled_ROIs32 * affinity_loss4).sum() / N32
        
        if self.loss_index == 0:
            affinity_loss = affinity_loss1
        elif self.loss_index == 1:
            affinity_loss = affinity_loss1 + affinity_loss2
        elif self.loss_index == 2:
            affinity_loss = affinity_loss1 + affinity_loss2 + affinity_loss3
        elif self.loss_index == 3:
            affinity_loss = affinity_loss1 + affinity_loss2 + affinity_loss3 + affinity_loss4
        else:
            affinity_loss = torch.zeros(1, dtype=seg.dtype, device=seg.device)
        return affinity_loss,pusdo_label

