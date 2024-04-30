import torch
import torch.nn as nn
import torch.nn.functional as F

class NLL(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(NLL, self).__init__()

    def forward(self, inputs, targets):
        #print(inputs)
        #print(targets)
        #comment out if your model contains a sigmoid or equivalent activation layer
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Calculate negative log likelihood loss
        loss = nn.NLLLoss()(log_probs, targets)
        
        return loss
    
class KLDiv(nn.Module):
    def __init__(self):
        super(KLDiv, self).__init__()

    def forward(self, input, target):
        #print(input)
        #print(target)
        # Ensure input is log probabilities
        input_log_probs = F.log_softmax(input, dim=1)
        
        # Ensure target is probabilities
        target_probs = F.one_hot(target, num_classes=input.size(1)).float()  # Adjust for 0-based indexing
        
        # Calculate KL Divergence loss
        loss = nn.KLDivLoss(reduction='batchmean')(input_log_probs, target_probs)
        
        return loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class CrossEntropy(nn.Module):
      def __init__(self, weight=None, size_average=True):
        super(CrossEntropy, self).__init__()
      def forward(self, inputs, targets, smooth=1):
        
        targets = torch.LongTensor(targets)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        CEL = F.cross_entropy(inputs, targets, reduction='mean')

        return CEL

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
    
class BCElossFuntion(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCElossFuntion, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        

        return BCE