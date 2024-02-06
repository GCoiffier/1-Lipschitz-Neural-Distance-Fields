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
    

def vector_alignment_loss(y, target):
    return (1-F.cosine_similarity(y, target, dim = 1)).mean()