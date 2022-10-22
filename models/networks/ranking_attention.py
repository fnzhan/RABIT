import math
import torch
from torch import nn
from operator import mul
from math import gcd
import torch.nn.functional as F
from inspect import isfunction
from functools import partial, wraps, reduce
from PIL import Image
# helper functions

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
    Gamma = u*G*v
    return Gamma


def ot_topk(logit, k, epsilon=5e-2, max_iter=10):
    # logit.shape=n, score
    logit = 1 - logit
    anchors = torch.FloatTensor([0, 1]).view([1, 1, 2])
    if torch.cuda.is_available():
        anchors = anchors.cuda()
    bs, n = logit.size()
    logit = logit.view([bs, n, 1])

    scores_ = logit.clone().detach()
    max_scores = torch.max(scores_).detach()
    scores_[scores_==float('-inf')] = float('inf')
    min_scores = torch.min(scores_).detach()
    filled_value = min_scores - (max_scores-min_scores)
    mask = logit==float('-inf')
    scores = logit.masked_fill(mask, filled_value)

    C = (scores-anchors)**2
    C = C / (C.max().detach())
    mu = torch.ones([bs, n, 1], requires_grad=False)/n
    nu = torch.FloatTensor([k/n, (n-k)/n]).view([1, 1, 2])
    nu = nu.repeat(bs, 1, 1)

    if torch.cuda.is_available():
        mu = mu.cuda()
        nu = nu.cuda()

    Gamma = sinkhorn_forward(C, mu, nu, epsilon, max_iter)
    A = Gamma[:,:,0] * n
    return A


def differentiable_topk(logit, k):
    # logit.shape=b,i,j
    x = logit.clone()

    index_ls = []
    values, indices = x.topk(k, dim=-1)
    index_ls.append(indices.unsqueeze(-1))

    values_sum = values.sum(-1, keepdim=True)
    topk_tensors = torch.zeros_like(x).scatter_(-1, indices, values)

    ret = topk_tensors / values_sum - logit.detach() + logit
    return ret



def differentiable_topk_old(x, k, temperature=1.):
    *_, n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        topks = (x * 100).softmax(dim=-1)
        topk_tensors.append(topks.unsqueeze(-1))
        values, indices = topks.topk(1, dim=-1)
        if not is_last:
            x.scatter_(-1, indices, float('-inf'))
    topks = torch.cat(topk_tensors, dim=-1)
    return topks

class RankNet(nn.Module):
    def __init__(self, patch_size, temperature, opt):
        super().__init__()
        self.patch_size = patch_size
        self.temperature = temperature
        self.opt = opt

    def forward(self, q, k, topk=1):
        b, N, D = q.shape
        w = int(N ** (1 / 2))
        stride = int(self.patch_size ** (1 / 2))

        b_q = unfold_bucket(self.patch_size, q)
        b_k = unfold_bucket(self.patch_size, k)  # 1, 4096, 4, 256
        sq = b_q.mean(dim=2).permute(0,2,1)
        sk = b_k.mean(dim=2).permute(0,2,1)

        # q = q.view(-1, w, w, D).permute(0, 3, 1, 2)
        # k = k.view(-1, w, w, D).permute(0, 3, 1, 2)
        # sq = F.unfold(q, kernel_size=self.opt.match_kernel, padding=int(
        #     self.opt.match_kernel // 2), stride=stride)
        # sk = F.unfold(k, kernel_size=self.opt.match_kernel, padding=int(
        #     self.opt.match_kernel // 2), stride=stride)

        Corr = torch.einsum('bei,bej->bij', sq, sk)
        # R_ = torch.matmul(sq, sk.permute(0, 2, 1))

        # R = F.softmax(Corr * 1000, dim=-1).unsqueeze(-1)
        # R_ = F.softmax(Corr.transpose(1, 2) * 1000, dim=-1).unsqueeze(-1)
        R = differentiable_topk(Corr, k=topk)
        R_ = differentiable_topk(Corr.transpose(1, 2), k=topk)

        return R, R_


def unfold_bucket(patch_size, input):
    b, N, D = input.shape  # _, 16384, 256
    w = int(N ** (1 / 2))
    patch_w = int(patch_size**(1/2))
    stride = patch_w
    n_patch = int(w / patch_w)

    input = input.view(-1, w, w, D).permute(0, 3, 1, 2) #_, 256, 128, 128
    output = F.unfold(input, kernel_size=patch_w, stride=stride) #_, 256*2*2, 64 * 64
    output = output.view(b, D, patch_w*patch_w, n_patch*n_patch).\
        permute(0, 3, 2, 1) # _, 4096, 4, 256
    return output

def fold_bucket(input):
    b, n_patch, patch_size, D = input.shape  # 3, 256, 64, 3
    input = input.permute(0, 3, 2, 1).contiguous().view(b, -1, n_patch)
    patch_w = int(patch_size ** (1 / 2))
    w = int((patch_size*n_patch) ** (1 / 2))
    stride = patch_w
    output = F.fold(input, output_size=(w, w), kernel_size=(patch_w, patch_w), stride=stride)
    output = output.permute(0, 2, 3, 1)

    return output


def align_feature(R, dots, v):
    _, bucket_num, patch_size, _ = dots.shape
    b, N, D = v.shape
    b_v = unfold_bucket(patch_size, v)
    b_v_r = torch.einsum('buvk,bvtd->butkd', R, b_v)

    b_v_r = b_v_r.reshape(b, bucket_num, -1, D)
    b_v = b_v_r  # [2, 4096, 12, 3]     dots: [2, 4096, 4, 12]
    # out = b_v[:, :, 0, :]

    out = torch.einsum('buij,buje->buie', dots, b_v)
    out = fold_bucket(out)
    out = out.reshape(b, N, D)

    # im = out[0].view(64, 64, 3)
    # im = im.cpu().detach().numpy()
    # im = ((im + 1.0) * 128).astype('uint8')
    # im = Image.fromarray(im).resize((256, 256))
    # im.save('models/networks/warp.png')
    #
    # im = v[0].view(64, 64, 3)
    # im = im.cpu().detach().numpy()
    # im = ((im + 1.0) * 128).astype('uint8')
    # im = Image.fromarray(im).resize((256, 256))
    # im.save('models/networks/exemplar.png')

    # print(1 / 0)
    return out



def align_feature_v(R, dots, v):
    _, patch_num, _, patch_size = dots.shape
    b, N, D = v.shape
    # R 4096, 4096, 3   , dots 4096, 12, 4

    b_v = unfold_bucket(patch_size, v)    # # _, 4096, 4, 3

    dots = dots.view(b, patch_num, patch_size, -1, patch_size)

    # print(dots.shape, R.shape)
    # print(1/0)

    b_v_l = torch.einsum('buikj,bujd->buikd', dots, b_v)  # 2, 4096, 4, 3, 256
    b_v_r = torch.einsum('buvk,bvikd->buid', R, b_v_l)  # 2, 4096, 4, 256

    # b_v_r = torch.einsum('buvk,bvtd->butkd', R, b_v)
    # b_v_r = b_v_r.reshape(b, bucket_num, -1, D)
    # b_v = b_v_r  # [2, 4096, 12, 3]     dots: [2, 4096, 4, 12]
    # out = b_v[:, :, 0, :]
    # b_v = b_v_r[:, :, :, 0, :]

    # out = torch.einsum('buij,bujd->buid', dots, b_v)
    out = fold_bucket(b_v_r)
    out = out.reshape(b, N, D)

    # im = out[0].view(128, 128, 3)
    # # im = im[:, :, 0, :]
    # im = im.cpu().detach().numpy()
    # im = ((im + 1.0) * 128).astype('uint8')
    # im = Image.fromarray(im)
    # im.save('models/networks/warp_cycle.png')
    # print(1 / 0)
    return out


class RAS(nn.Module):
    def __init__(self, patch_size, temperature=0.75, n_top=1, opt=None):
        super().__init__()
        self.patch_size = patch_size
        self.temperature = temperature
        self.ranknet = RankNet(patch_size=self.patch_size, temperature=temperature, opt=opt)
        self.n_top = n_top
        self.opt = opt

    def forward(self, q, k, R):
        b, N, D = q.shape
        buckets = N // self.patch_size

        # R, R_v = self.ranknet(q, k, topk=self.n_top)
        # R = R.type_as(q).to(q)  # [3, 256, 256]

        b_q = unfold_bucket(self.patch_size, q)
        b_k = unfold_bucket(self.patch_size, k)
        # print (R.shape, b_k.shape)

        b_k_r = torch.einsum('buvk,bvtd->butkd', R, b_k)  #  2, 4096, 4, 3, 256
        b_k_r = b_k_r.reshape(b, buckets, self.patch_size, self.n_top, D)

        dots_ = torch.einsum('buid,bujkd->buijk', b_q, b_k_r)
        dots_ = dots_.view(b, buckets, self.patch_size, -1)  # 2, 4096, 4, 12
        dots = (dots_ * 1000).softmax(dim=-1)
        dots_v = (dots_.transpose(2, 3) * 1000).softmax(dim=-1) # 2, 4096, 12, 4
        # dots = dots_.view(b, buckets, self.patch_size, -1)

        # R: [2, 4096, 4096, 3], dots: [2, 4096, 4, 12]
        return dots, dots_v
