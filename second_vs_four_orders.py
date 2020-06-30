import torch
from utils import *
from lie_group_utils import SO3, SE3_2
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
torch.set_default_dtype(torch.float64)
from preintegration_utils import *



def propagate(T0, P, Upsilon, Q, method, dt, g, cholQ=0):
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

    Pprime = axat(AdUps.mm(F), P)
    # compound the covariances based on the second-order method
    Pprop = Pprime + Q

    if method == 1:
        # add fourth-order method
        Pprop += four_order(Pprime, Q)

    elif method == 2:
        # Monte Carlo method
        n_tot_samples = 1000000
        nsamples = 50000
        N = int(n_tot_samples/nsamples)+1

        tmp = torch.cholesky(P + 1e-20*torch.eye(9))
        cholP = tmp.cuda().expand(nsamples, 9, 9)
        cholQ = cholQ.cuda().expand(nsamples, 9, 9)

        Pprop = torch.zeros(9, 9)
        
        Gamma = Gamma.cuda().expand(nsamples, 5, 5)
        Upsilon = Upsilon.cuda().expand(nsamples, 5, 5)
        T0 = T0.cuda().expand(nsamples, 5, 5)
        T_inv = T.inverse().cuda().expand(nsamples, 5, 5)
        for i in range(N):
            xi0 = bmv(cholP, torch.randn(nsamples, 9).cuda())
            w = bmv(cholQ, torch.randn(nsamples, 9).cuda())    
            T0_i = T0.bmm(SE3_2.exp(xi0))
            Phi = f_flux(T0_i, dt)
            Upsilon_i = Upsilon.bmm(SE3_2.exp(w))
            T_i = Gamma.bmm(Phi).bmm(Upsilon_i)

            xi = SE3_2.log(T_inv.bmm(T_i))
            xi_mean = xi.mean(dim=0)
            Pprop += bouter(xi-xi_mean, xi-xi_mean).sum(dim=0).cpu()

        Pprop = Pprop / (N*nsamples+1)

    Pprop = (Pprop + Pprop.t())/2 # symmetric
    return T, Pprop

def plot_ellipse_SE32(T_est, P_est, v, color, name, i1, i2):
    """
    Draw ellipse based on the 3 more important directions of the covariance
    """
    D, V = torch.eig(P_est, eigenvectors=True)
    Y, I = torch.sort(D[:, 0], descending=True)
    a = 3*D[I[0], 0].sqrt() * V[:, I[0]]
    b = 3*D[I[1], 0].sqrt() * V[:, I[1]]
    c = 3*D[I[2], 0].sqrt() * V[:, I[2]]
    for n in range(3):
        if n == 0:
            xi = a*v.sin() + b*v.cos()
        elif n == 1:
            xi = b*v.sin() + c*v.cos()
        elif n == 2:
            xi = a*v.sin() + c*v.cos()
        Ttemp = T_est[-1].expand(50, 5, 5).bmm(SE3_2.exp(xi.cuda()).cpu())
        clines = Ttemp[:, :3, 4]
        if n == 0:
            plt.plot(clines[:, i1], clines[:, i2], label=name, color=color)
        else:
            plt.plot(clines[:, i1], clines[:, i2], color=color)


def main(i_max, k_max, T0, P0, Upsilon, Q, cholQ, dt, g):
    # Generate some random samples
    T = torch.zeros(i_max, k_max, 5, 5).cuda()
    T[:, 0] = T0.cuda().repeat(i_max, 1, 1)
    tmp = P0.sqrt().cuda().expand(i_max, 9, 9) # Pxi assumed diagonal!
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
    P_est1 = torch.zeros(k_max, 9, 9) # second order covariance
    P_est2 = torch.zeros(k_max, 9, 9) # fourth order covariance
    P_est_mc = torch.zeros(k_max, 9, 9) # SO(3) x R^6 covariance
    T_est[0] = T0
    P_est1[0] = P0.clone()
    P_est2[0] = P0.clone()
    P_est_mc[0] = P0.clone()
    for k in range(1, k_max):
        # Second-order method
        T_est[k], P_est1[k] = propagate(T_est[k-1], P_est1[k-1], Upsilon, Q, 0, dt, g)
        # Fourth-order method
        _, P_est2[k] = propagate(T_est[k-1], P_est2[k-1], Upsilon, Q, 1, dt, g)
        # baseline method
        _, P_est_mc[k] = propagate(T_est[k-1], P_est_mc[k-1], Upsilon, Q, 2, dt, g, cholQ)
        
    res = torch.zeros(3)
    res[1] = fro_norm(P_est_mc[-1], P_est1[-1])
    res[2] = fro_norm(P_est_mc[-1], P_est2[-1])
    print(fro_norm(P_est1[-1], P_est2[-1]))
    return res

if __name__ == '__main__':
    ### Parameters ###
    i_max = 5000 # number of random points
    k_max = 301 # number of compounded poses
    g = torch.Tensor([0, 0, 9.81]) # gravity vector
    dt = 0.05 # step time (s)
    sigmas = 0.03*torch.Tensor([0.1, 0.5, 1, 2, 3, 4, 5])
    # Constant acceleration, noise on IMU
    # Define a PDF over transformations (mean and covariance)
    xibar = torch.Tensor([0, 0, 0, 1, 0, 0, 0, 0, 0]).cuda()*dt
    Upsilon = SE3_2.uexp(xibar).cpu()
    Upsilon[:3, 3] += -g*dt
    Upsilon[:3, 4] = Upsilon[:3, 3]*dt/2

    T0 = torch.eye(5)
    P0 = torch.zeros(9, 9)
    res = torch.zeros(sigmas.shape[0], 3)
    for i in range(sigmas.shape[0]):
        # Define right perturbation noise
        cholQ = torch.Tensor([0, 0, sigmas[i], 0, 0, 0, 0, 0, 0]).diag()
        Q = cholQ.mm(cholQ.t())
        res[i] = main(i_max, k_max, T0, P0, Upsilon, Q, cholQ, dt, g)
        print(i, res[i])
    res[:, 0] = sigmas
    np.savetxt('figures/second_vs_four_order.txt', res.numpy(), comments="", header="sigma second four")
