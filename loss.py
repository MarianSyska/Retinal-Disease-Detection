import torch
from torch import nn
from torch.functional import F

class FocalRetinalDiseaseLoss(nn.Module):
    
    def __init__(self, gamma, alpha_retino, alpha_edema):
        super(FocalRetinalDiseaseLoss, self).__init__()
        
        self.loss_fn_retino = MultiClassFocalLoss(gamma=gamma, alpha=alpha_retino, num_classes=5)
        self.loss_fn_edema = BinaryFocalLoss(gamma=gamma, alpha=alpha_edema)
    
    def forward(self, input, target):
        return self.loss_fn_retino(input[:,:5], target[:,0]) + self.loss_fn_edema(input[:,5], target[:,1])


class NormalRetinalDiseaseLoss(nn.Module):
    
    def __init__(self, alpha_retino=None, alpha_edema=None):
        super(NormalRetinalDiseaseLoss, self).__init__()
        self.loss_fn_retino = nn.CrossEntropyLoss(weight=alpha_retino)
        self.loss_fn_edema = nn.BCEWithLogitsLoss(weight=alpha_edema)
    
    def forward(self, input, target):
        return self.loss_fn_retino(input[:,:5], target[:,0]) + self.loss_fn_edema(input[:,5], target[:,1].to(dtype=torch.float32))


class MultiClassFocalLoss(nn.Module):
    
    def __init__(self, gamma, alpha, num_classes):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes


    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        pt = 0.0
        targets = targets.long()
        inputs_sm = F.softmax(inputs, dim=1)
        selector = nn.functional.one_hot(targets, num_classes=self.num_classes).bool()
        pt = inputs_sm[selector]
        
        F_loss = (1 - pt)**self.gamma * ce
        
        return F_loss.mean()
    

class BinaryFocalLoss(nn.Module):
    
    def __init__(self, gamma, alpha):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha


    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets.to(torch.float32), reduction='none')
        
        pt = torch.exp(-bce)
                        
        F_loss = self.alpha * (1 - pt)**self.gamma * bce
        
        return F_loss.mean()