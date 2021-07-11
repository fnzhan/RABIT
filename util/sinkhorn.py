# -*- coding: utf-8 -*-
import math
import torch


def sinkhorn(dot, mask=None, eps=1e-100, return_kernel=False, max_iter=100):
    """
    dot: n x in_size x out_size
    mask: n x in_size
    output: n x in_size x out_size
    """
    n, in_size, out_size = dot.shape
    if return_kernel:
        K = torch.exp(dot / eps)
    else:
        K = dot

    mu = torch.ones([n, out_size], requires_grad=False) / out_size
    nu = torch.FloatTensor([[0.6], [0.4]]).view([1, 2])

    v = K.new_ones((n, out_size)) / 5
    # print('********') input: 2, output: 5
    a = float(out_size / in_size)
    for _ in range(max_iter):
        u = a * nu / torch.bmm(K, v.view(n, out_size, 1)).view(n, in_size)
        if mask is not None:
            u = u * mask
        v = mu / torch.bmm(u.view(n, 1, in_size), K).view(n, out_size)
    K = u.view(n, in_size, 1) * (K * v.view(n, 1, out_size))
    if return_kernel:
        K = K / out_size
        return (K * dot).sum(dim=[1, 2])
    return K

def log_sinkhorn(K, mask=None, eps=1.0, return_kernel=False, max_iter=100):
    """
    dot: n x in_size x out_size
    mask: n x in_size
    output: n x in_size x out_size
    """
    batch_size, in_size, out_size = K.shape
    def min_eps(u, v, dim):
        Z = (K + u.view(batch_size, in_size, 1) + v.view(batch_size, 1, out_size)) / eps
        return -torch.logsumexp(Z, dim=dim)
    # K: batch_size x in_size x out_size
    u = K.new_zeros((batch_size, in_size))
    v = K.new_zeros((batch_size, out_size))
    a = torch.ones_like(u).fill_(out_size / in_size)
    if mask is not None:
        a = out_size / mask.float().sum(1, keepdim=True)
    a = torch.log(a)
    for _ in range(max_iter):
        u = eps * (a + min_eps(u, v, dim=-1)) + u
        if mask is not None:
            u = u.masked_fill(~mask, -1e8)
        v = eps * min_eps(u, v, dim=1) + v
    if return_kernel:
        output = torch.exp(
            (K + u.view(batch_size, in_size, 1) + v.view(batch_size, 1, out_size)) / eps)
        output = output / out_size
        return (output * K).sum(dim=[1, 2])
    K = torch.exp(
        (K + u.view(batch_size, in_size, 1) + v.view(batch_size, 1, out_size)) / eps)
    return K

def multihead_attn(input, weight, mask=None, eps=1.0, return_kernel=False,
                   max_iter=100, log_domain=False, position_filter=None):
    """Comput the attention weight using Sinkhorn OT
    input: n x in_size x in_dim
    mask: n x in_size
    weight: m x out_size x in_dim (m: number of heads/ref)
    output: n x out_size x m x in_size
    """
    n, in_size, in_dim = input.shape
    m, out_size = weight.shape[:-1]
    # K = torch.tensordot(input, weight, dims=[[-1], [-1]])
    K = torch.einsum('bid,bod->bio', input, weight)

    # K = K.permute(0, 2, 1, 3)
    # if position_filter is not None:
    #     K = position_filter * K
    # K: n x m x in_size x out_size
    K = K.reshape(-1, in_size, out_size)
    # K: nm x in_size x out_size
    if mask is not None:
        mask = mask.repeat_interleave(m, dim=0)
    if log_domain:
        K = log_sinkhorn(K, mask, eps, return_kernel=return_kernel, max_iter=max_iter)
    else:
        if not return_kernel:
            K = torch.exp(K / eps)
        K = sinkhorn(K, mask, eps, return_kernel=return_kernel, max_iter=max_iter)
    # K: nm x in_size x out_size
    if return_kernel:
        return K.reshape(n, m)
    # K = K.reshape(n, m, in_size, out_size)
    # if position_filter is not None:
    #     K = position_filter * K
    K = K.permute(0, 2, 1).contiguous()
    return K

# def sinkhorn(dot, mask=None, eps=1e-03, return_kernel=False, max_iter=100):
#     """
#     dot: n x in_size x out_size
#     mask: n x in_size
#     output: n x in_size x out_size
#     """
#     n, in_size, out_size = dot.shape
#     if return_kernel:
#         K = torch.exp(dot / eps)
#     else:
#         K = dot
#     # K: n x in_size x out_size
#     u = K.new_ones((n, in_size))
#     v = K.new_ones((n, out_size))
#     a = float(out_size / in_size)
#     if mask is not None:
#         mask = mask.float()
#         a = out_size / mask.sum(1, keepdim=True)
#     for _ in range(max_iter):
#         u = a / torch.bmm(K, v.view(n, out_size, 1)).view(n, in_size)
#         if mask is not None:
#             u = u * mask
#         v = 1. / torch.bmm(u.view(n, 1, in_size), K).view(n, out_size)
#     K = u.view(n, in_size, 1) * (K * v.view(n, 1, out_size))
#     if return_kernel:
#         K = K / out_size
#         return (K * dot).sum(dim=[1, 2])
#     return K