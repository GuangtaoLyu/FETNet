import torch
import torch.nn as nn
import torchvision.models as models


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss
        
def l1_loss(f1, f2, mask=1):
    return torch.mean(torch.abs(f1 - f2) * mask)

def style_loss(A_feats, B_feats):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        _, c, w, h = A_feat.size()
        A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
        B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
        A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
        B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
        loss_value = loss_value+torch.mean(torch.abs(A_style - B_style) / (c * w * h))
    return loss_value

def TV_loss(x):
    h_x = x.size(2)
    w_x = x.size(3)
    h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]))
    w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]))
    return h_tv + w_tv

def preceptual_loss(A_feats, B_feats):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        loss_value = loss_value + torch.mean(torch.abs(A_feat - B_feat))

    return loss_value
   
def diceLoss(predict,target):
    r"""
    Dice loss
    https:
    """
    epsilon = 1e-5
    assert predict.size() == target.size(), "the size of predict and target must be equal."
    num = predict.size(0)
    pre = predict.view(num, -1)
    tar = target.view(num, -1)
    intersection = (pre * tar).sum(-1).sum() 
    union = (pre + tar).sum(-1).sum()
    score = 1 - 2 * (intersection + epsilon) / (union + epsilon)
    return score


