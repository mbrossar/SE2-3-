import torch
from utils import *
from lie_group_utils import SO3, SE3_2
import matplotlib.pyplot as plt
import numpy as np
import pickle
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=4)


def f_Gamma(g, dt):
    """Compute Gamma preintegration"""
    Gamma = torch.eye(5)
    Gamma[:3, 3] = g*dt
    Gamma[:3, 4] = 1/2*g*(dt**2)
    return Gamma

def Phi(T, dt):
    """Compute Phi (the flux) preintegration"""
    if dt <= 0:
        return T
    T = T.clone()
    T[:3, 4] += T[:3, 3]*dt
    return T

i_max = 10
dt = 0.07
g = torch.tensor([0, 0, 9.81])
Gamma = f_Gamma(g, dt)
Ups_i = SE3_2.exp(torch.randn(i_max+1, 9).cuda()).cpu()
Adj_i = SE3_2.Ad(Ups_i.inverse())
F = torch.eye(9)
F[6:9, 3:6] = dt*torch.eye(3)

# check eq. (53) and (54)
# incremental computation
Gamma_incr = torch.eye(5)
Ups_incr = torch.eye(5)
for i in range(i_max+1):
    Gamma_incr = Gamma.mm(Phi(Gamma_incr, dt))
    Ups_incr = Phi(Ups_incr, dt).mm(Ups_i[i])
# batch computation
Gamma_b = torch.eye(5)
Ups_b = torch.eye(5)
i = 0
j = i_max+1
for k in range(i, j):
    Ups_b = Ups_b.mm(Phi(Ups_i[k], (j-1-k)*dt))
    Gamma_b = Gamma_b.mm(Phi(Gamma, (k-1)*dt))
# print(Ups_b)
# print(Ups_incr)
# print(Gamma_b)
# print(Gamma_incr)

# check eq. (55) without noise
T_est = SE3_2.uexp(torch.randn(9).cuda()).cpu()
xi0 = 0.05*torch.randn(9)
xi_est = xi0.clone()
T = T_est.mm(SE3_2.uexp(xi0.cuda()).cpu())
for i in range(i_max+1):
    T = Gamma.mm(Phi(T, dt)).mm(Ups_i[i])
    T_est = Gamma.mm(Phi(T_est, dt)).mm(Ups_i[i])
    A_i = Adj_i[i].mm(F)
    xi_est = A_i.mv(xi_est)

xi_final = SE3_2.ulog(T_est.inverse().mm(T).cuda()).cpu()

A_i_0 = torch.eye(9)
for l in range(j):
    A_i = Adj_i[l].mm(F)
    A_i_0 = A_i.mm(A_i_0)
xi_batch_est = A_i_0.mv(xi0)
# print(xi_final)
# print(xi_est)
# print(xi_batch_est)


# check eq. (55) with noise
T_est = SE3_2.uexp(torch.randn(9).cuda()).cpu()
xi0 = 0.05*torch.randn(9)
eta_i = 0.05*torch.randn(i_max+1, 9)

tmp = SE3_2.exp(eta_i.cuda()).cpu()
Adj_i = SE3_2.Ad((Ups_i.bmm(tmp)).inverse())

xi_est = xi0.clone()
T = T_est.mm(SE3_2.uexp(xi0.cuda()).cpu())
for i in range(i_max+1):
    T = Gamma.mm(Phi(T, dt)).mm(Ups_i[i])
    tmp = SE3_2.uexp(eta_i[i].cuda()).cpu()
    T_est = Gamma.mm(Phi(T_est, dt)).mm(Ups_i[i].mm(tmp))
    A_i = Adj_i[i].mm(F)
    T_est.mm(SE3_2.uexp(A_i.mv(xi_est).cuda()).cpu().mm(tmp.inverse()))
    xi_est = SE3_2.ulog(SE3_2.uexp(A_i.mv(xi_est).cuda()).mm(tmp.inverse().cuda())).cpu()

xi_final = SE3_2.ulog(T_est.inverse().mm(T).cuda()).cpu()

A_i_0 = torch.eye(9)
for l in range(j):
    A_i = Adj_i[l].mm(F)
    A_i_0 = A_i.mm(A_i_0)
xi_batch_est = A_i_0.mv(xi0)
T_err = SE3_2.uexp(xi_batch_est.cuda()).cpu()
for k in range(i_max + 1):
    A_i_k_1 = torch.eye(9)
    for l in range(k+1, i_max+1):
        A_i = Adj_i[l].mm(F)
        A_i_k_1 = A_i.mm(A_i_k_1)
    Eta_k = SE3_2.uexp((A_i_k_1.mv(-eta_i[k])).cuda()).cpu()
    T_err = T_err.mm(Eta_k)

xi_batch_est = SE3_2.ulog(T_err.cuda()).cpu()
print(xi_final)
print(xi_est)
print(xi_batch_est)
