import numpy as np
import torch
from torch.autograd import Function
import torch.nn.functional as F

def sinkhorn_forward(C, mu, nu, epsilon, max_iter):
    """standard forward of sinkhorn."""

    bs, _, k_ = C.size()

    v = torch.ones([bs, 1, k_])/(k_)
    G = torch.exp(-C/epsilon)
    if torch.cuda.is_available():
        v = v.cuda()

    for _ in range(max_iter):
        u = mu/(G*v).sum(-1, keepdim=True)
        v = nu/(G*u).sum(-2, keepdim=True)

    # print(G.shape)
    Gamma = u*G*v
    # print(Gamma.shape)
    return Gamma


class TopK_custom(torch.nn.Module):
    def __init__(self, k, epsilon=0.1, max_iter=50):
        super(TopK_custom, self).__init__()
        self.k = k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([0,1]).view([1,1, 2])
        self.max_iter = max_iter

        if torch.cuda.is_available():
            self.anchors = self.anchors.cuda()

    def forward(self, scores):
        print (scores.shape)
        bs, n = scores.size()
        scores = scores.view([bs, n, 1])

        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_==float('-inf')] = float('inf')
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores-min_scores)
        mask = scores==float('-inf')
        scores = scores.masked_fill(mask, filled_value)

        C = (scores-self.anchors)**2
        C = C / (C.max().detach())
        mu = torch.ones([bs, n, 1], requires_grad=False)/n
        nu = torch.FloatTensor([self.k/n, (n-self.k)/n]).view([1, 1, 2])
        nu = nu.repeat(bs, 1, 1)

        if torch.cuda.is_available():
            mu = mu.cuda()
            nu = nu.cuda()

        Gamma = sinkhorn_forward(C, mu, nu, self.epsilon, self.max_iter)
        A = Gamma[:,:,0]*n
        return A

############################################################################
############################################################################