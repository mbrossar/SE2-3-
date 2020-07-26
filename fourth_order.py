import torch
from utils import *
from lie_group_utils import SO3, SE3_2
import matplotlib.pyplot as plt
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'x-large'
plt.rcParams['legend.fontsize'] = 'x-large'
plt.rcParams['xtick.labelsize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'
plt.rcParams['text.usetex'] =True
params= {'text.latex.preamble' : [r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']}
plt.rcParams.update(params)
import numpy as np
torch.set_default_dtype(torch.float64)
from preintegration_utils import *


def propagate(T0, Sigma, Upsilon, Q, method, dt, g, cholQ=0):
    """Propagate state for one time step"""
    Gamma = f_Gamma(g, dt)
    Phi = f_flux(T0, dt)
    # propagate the mean
    T = Gamma.mm(Phi).mm(Upsilon)

    # Jacobian for propagating prior along time
    F = torch.eye(9)
    F[6:9, 3:6] = torch.eye(3)*dt

    # compute Adjoint of right transformation mean
    AdUps = SE3_2.uAd(SE3_2.uinv(Upsilon))

    Sigma_tmp = axat(AdUps.mm(F), Sigma)
    # compound the covariances based on the second-order method
    Sigma_prop = Sigma_tmp + Q

    if method == 1:
        # add fourth-order method
        Sigma_prop += four_order(Sigma_tmp, Q)

    Sigma_prop = (Sigma_prop + Sigma_prop.t())/2 # symmetric
    return T, Sigma_prop

def main(i_max, k_max, T0, Sigma0, Upsilon, Q, cholQ, dt, g):
    # Generate some random samples
    T = torch.zeros(i_max, k_max, 5, 5).cuda()
    T[:, 0] = T0.cuda().repeat(i_max, 1, 1)
    tmp = Sigma0.sqrt().cuda().expand(i_max, 9, 9) # Sigma0 assumed diagonal!
    T[:, 0] = T[:, 0].bmm(SE3_2.exp(bmv(tmp, torch.randn(i_max, 9).cuda())))
    Gamma = f_Gamma(g, dt).cuda().expand(i_max, 5, 5)
    tmp = cholQ.cuda().expand(i_max, 9, 9)
    tmp2 = Upsilon.cuda().expand(i_max, 5, 5)
    for k in range(1, k_max):
        T_k = SE3_2.exp(bmv(tmp, torch.randn(i_max, 9).cuda()))
        Phi = f_flux(T[:, k-1], dt)
        T[:, k] = Gamma.bmm(Phi).bmm(tmp2).bmm(T_k)
    T = T.cpu()

    # Propagate the uncertainty using second- and fourth-order methods
    T_est = torch.zeros(k_max, 5, 5)
    Sigma2th = torch.zeros(k_max, 9, 9) # second order covariance
    Sigma4th = torch.zeros(k_max, 9, 9) # fourth order covariance
    
    T_est[0] = T0
    Sigma2th[0] = Sigma0.clone()
    Sigma4th[0] = Sigma0.clone()
    for k in range(1, k_max):
        # Second-order method
        T_est[k], Sigma2th[k] = propagate(T_est[k-1], Sigma2th[k-1], Upsilon, Q, 0, dt, g)
        # Fourth-order method
        _, Sigma4th[k] = propagate(T_est[k-1], Sigma4th[k-1], Upsilon, Q, 1, dt, g)
        
    xi = SE3_2.log((T_est[-1].inverse().expand(i_max, 5, 5).bmm(T[:, -1])).cuda())
    P_est_mc = bouter(xi, xi).sum(dim=0).cpu()/(i_max-1)
    res = torch.zeros(3)
    res[1] = fro_norm(P_est_mc[-1], Sigma2th[-1])
    res[2] = fro_norm(P_est_mc[-1], Sigma4th[-1])
    return res

if __name__ == '__main__':
    path = 'figures/second_vs_four_order.txt'
    ### Parameters ###
    i_max = 5000 # number of random points
    k_max = 301 # number of compounded poses
    g = torch.Tensor([0, 0, 9.81]) # gravity vector
    dt = 0.05 # step time (s)
    sigmas = 0.03*torch.Tensor([0.1, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    # Constant acceleration, noise on IMU
    # Define a PDF over transformations (mean and covariance)
    xibar = torch.Tensor([0, 0, 0, 1, 0, 0, 0, 0, 0]).cuda()*dt
    Upsilon = SE3_2.uexp(xibar).cpu()
    Upsilon[:3, 3] += -g*dt
    Upsilon[:3, 4] = Upsilon[:3, 3]*dt/2

    T0 = torch.eye(5)
    Sigma0 = torch.zeros(9, 9)
    res = torch.zeros(sigmas.shape[0], 3)
    for i in range(sigmas.shape[0]):
        # Define right perturbation noise
        cholQ = torch.Tensor([0, 0, sigmas[i], 0, 0, 0, 0, 0, 0]).diag()
        Q = cholQ.mm(cholQ.t())
        res[i] = main(i_max, k_max, T0, Sigma0, Upsilon, Q, cholQ, dt, g)
    res[:, 0] = sigmas
    # np.savetxt(path, res.numpy(), comments="", header="sigma second four")
    
    plt.plot(res[:, 0], res[:, 2], color='cyan')
    plt.plot(res[:, 0], res[:, 1], color='green')
    plt.xlabel(r'propagation noise $\sigma$ (rad/s)')
    plt.ylabel(r'covariance error')
    plt.legend(["fourth-order", "second-order"])
    plt.grid()
    plt.xlim(0, sigmas[-1])
    plt.ylim(0)
    plt.show()
