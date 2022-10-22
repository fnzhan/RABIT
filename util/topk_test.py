import torch


# torch.manual_seed(1)
# num_iter = int(10)
# k = 1
# epsilon=5e-2 # larger epsilon lead to smoother relaxation, and requires less num_iter
# soft_topk = solver.TopK_custom(k, epsilon=epsilon, max_iter=num_iter)

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

scores = [[0.5, 0.2, 0.3, 0.4, 0.6], [0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.1, 0.2, 0.4, 0.5]]
scores_tensor = torch.Tensor(scores).type(torch.FloatTensor)
# A = ot_topk(scores_tensor, k=1)
A = differentiable_topk(scores_tensor, k=1)

indicator_vector = A.data.cpu().detach().numpy()
print('======topk scores======')
print(indicator_vector)
