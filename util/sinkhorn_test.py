import torch
from sinkhorn import multihead_attn
from sinkhorn_solver import SinkhornSolver
from torch.nn.parameter import Parameter
import topk_ranking as solver


torch.manual_seed(1)
num_iter = int(10)
k = 1
epsilon=5e-2 # larger epsilon lead to smoother relaxation, and requires less num_iter
soft_topk = solver.TopK_custom(k, epsilon=epsilon, max_iter=num_iter)

scores = [[0.5, 0.2, 0.3, 0.4, 0.6], [0.1, 0.2, 0.3, 0.4, 0.5]]
#input the scores here
# scores_tensor = Parameter(torch.FloatTensor([scores]))
# print('======scores======')
# print(scores)
scores_tensor = torch.Tensor(scores).type(torch.FloatTensor)
A = soft_topk(scores_tensor.cuda())
indicator_vector = A.data.cpu().detach().numpy()
print('======topk scores======')
print(indicator_vector)




# epsilon = 10**(-3)
# solver = SinkhornSolver(epsilon=epsilon, iterations=100)
# # sinkhornsolver = SinkhornSolver(10)
#
# x = torch.Tensor([[-1], [1]]).type(torch.FloatTensor)
# y = torch.Tensor([[-1], [-0.5], [0.1], [0.5], [1]]).type(torch.FloatTensor)
# # print(x.shape)
# # n = 5
# # x = torch.randn(n//2, 1) / 1.5
# # print (x.type(), x.shape)
# # print(x)
# # y = torch.randn(n, 1) - 2.0
#
# f, p = solver.forward(x, y)
#
# print(p.shape)
# print(p)



# x = torch.Tensor([[[-1], [1]]])
# y = torch.Tensor([[[-1], [-0.5], [0.0], [0.5], [1]]])
# f = multihead_attn(x, y, eps=0.05, max_iter=100, log_domain=False)
# print(f.shape)
# print(f)
