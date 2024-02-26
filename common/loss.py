import torch
import torch.nn.functional as F

class LossHKR:
    """ Hinge Kantorovitch-Rubinstein loss"""

    def __init__(self, margin, lbda):
        self.margin = margin  # must be small but not too small.
        self.lbda   = lbda  # must be high.

    def __call__(self, y):
        """
        Args:
            y: vector of predictions.
        """
        return  F.relu(self.margin - y) + (1./self.lbda) * torch.mean(-y)
        # return torch.mean(-y)
    
def vector_alignment_loss(y, target):
    return (1-F.cosine_similarity(y, target, dim = 1)*2).mean()

class SALLoss:
    """SAL: Sign Agnostic Learning of Shapes from Raw Data"""
    def __init__(self, l=1., metric="l2"):
        self.l = l

        self.callfun = {
            "l2" : self.SAL_l2,
            "l0" : self.SAL_l0
        }.get(metric, self.SAL_l2)

    def __call__(self, y_pred, y_target):
        return self.callfun(y_pred, y_target)

    def SAL_l2(self,y_pred,y_target):
        return torch.mean(torch.abs(torch.abs(y_pred) - y_target)**self.l)
    
    def SAL_l0(self,y_pred,y_target):
        return torch.mean(torch.abs(torch.abs(y_pred) - 1)**self.l)
    

class SALDLoss:
    """
    SALD: Sign Agnostic Learning with Derivatives
    
    Improving SAL using losses on gradients
    """
    def __init__(self):
        pass

    def __call__(self,y_pred,y_target):
        return torch.min( torch.norm(y_pred-y_target), torch.norm(y_pred+y_target))