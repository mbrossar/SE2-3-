import torch
from utils import *
from lie_group_utils import SO3, SE3_2
from preintegration_utils import *
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
import scipy.linalg
torch.set_default_dtype(torch.float64)


def propagate(T0, P, Upsilon, Q, method, dt, g, cholQ=0):
    """
    Propagate an extended pose and its uncertainty
    """
    Gamma = f_Gamma(g, dt)
    Phi = f_flux(T0, dt)
    # compound the mean
    T = Gamma.mm(Phi).mm(Upsilon)

    # Jacobian for propagating prior along time
    F = torch.eye(9)
    F[6:9, 3:6] = torch.eye(3)*dt

    # compute Adjoint of right transformation mean
    AdUps = SE3_2.uAd(SE3_2.uinv(Upsilon))
    Pprime = axat(AdUps.mm(F), P)
    # compound the covariances based on the second-order method
    Pprop = Pprime + Q

    if method == 2:
        # baseline SO(3) x R^6
        wedge_acc = SO3.uwedge(Upsilon[:3, 3]) # already multiplied by dt
        F = torch.eye(9)
        F[3:6, :3] = T0[:3, :3].t()
        F[3:6, :3] = -T0[:3, :3].mm(wedge_acc)
        F[6:9, :3] = F[3:6, :3]*dt/2
        F[6:9, 3:6] = dt*torch.eye(3)

        G = torch.zeros(9, 6)
        G[:3, :3] = dt*T0[:3, :3].t()
        G[3:6, 3:6] = T0[:3, :3]*dt
        G[6:9, 3:6] = 1/2*T0[:3, :3]*(dt**2)
        Pprop = axat(F, P) + axat(G, Q[:6, :6]/(dt**2))

    Pprop = (Pprop + Pprop.t())/2
    return T, Pprop

def plot_se23_helper(T_est, P_est, color, i1, i2, i_max):
    P_est_chol = torch.cholesky(P_est + torch.eye(9)*1e-16)
    r = bmv(P_est_chol.expand(i_max, 9, 9), torch.randn(i_max, 9))
    Ttemp = T_est[-1].expand(i_max, 5, 5).bmm(SE3_2.exp(r.cuda()).cpu())
    p_est = Ttemp[:, :3, 4]
    plt.scatter(p_est[:, i1], p_est[:, i2], color=color, s=3)
    if i1 == 0 and i2 == 1:
        np.savetxt("figures/intro_est.txt", p_est.numpy(), header="x y z", comments='')

def plot_so3_helper(T_est, P_est, color, i1, i2, i_max):
    r = torch.randn(i_max, 3)
    P_est_chol = torch.cholesky(P_est[6:9, 6:9]+torch.eye(3)*1e-16)
    p_est = bmv(P_est_chol.expand(i_max, 3, 3), r) + T_est[-1, :3, 4]
    plt.scatter(p_est[:, i1], p_est[:, i2], color=color, s=2, alpha=0.5)
    if i1 == 0 and i2 == 1:
        np.savetxt("figures/intro_b.txt", p_est.numpy(), header="x y z", comments='')

def main(i_max, k_max, T0, P0, Upsilon, Q, cholQ, dt, g, sigma, m_max):
    # Generate some random samples
    T = torch.zeros(i_max, k_max, 5, 5).cuda()
    T[:, 0] = T0.cuda().repeat(i_max, 1, 1)
    # NOTE: no initial uncertainty
    Gamma = f_Gamma(g, dt).cuda().expand(i_max, 5, 5)
    tmp = cholQ.cuda().expand(i_max, 9, 9)
    tmp2 = Upsilon.cuda().expand(i_max, 5, 5)
    for k in range(1, k_max):
        T_k = SE3_2.exp(bmv(tmp, torch.randn(i_max, 9).cuda()))
        Phi = f_flux(T[:, k-1], dt)
        T[:, k] = Gamma.bmm(Phi).bmm(T_k).bmm(tmp2)
    T = T.cpu()

    # Propagate the uncertainty methods
    T_est = torch.zeros(k_max, 5, 5)
    P_est = torch.zeros(k_max, 9, 9)
    P_est_b = torch.zeros(k_max, 9, 9) # SO(3) x R^6 covariance
    T_est[0] = T0
    P_est[0] = P0.clone()
    P_est_b[0] = P0.clone()
    for k in range(1, k_max):
        T_est[k], P_est[k] = propagate(T_est[k-1], P_est[k-1], Upsilon, Q, 1, dt, g)
        # baseline method
        _, P_est_b[k] = propagate(T_est[k-1], P_est_b[k-1], Upsilon, Q, 2, dt, g)

    # Now plot the transformations
    labels = ['x (m)', 'y (m)', 'z (m)']
    for i1, i2 in ((0, 1), (0, 2), (1, 2)):
        plt.figure()
        # Plot the covariance of the samples
        plot_so3_helper(T_est, P_est_b[-1], "red", i1, i2, i_max)
        # Plot the propagated covariance projected onto i1, i2
        plot_se23_helper(T_est, P_est[-1], 'green', i1, i2, i_max)
        # Plot the random samples' xy-locations
        plt.scatter(T[:, -1, i1, 4], T[:, -1, i2, 4], s=1, color='black', alpha=0.5)
        plt.scatter(T_est[-1, i1, 4], T_est[-1, i2, 4], color='yellow', s=30)
        plt.xlabel(labels[i1])
        plt.ylabel(labels[i2])
        plt.legend([r"$SO(3) \times \mathbb{R}^6$", "$SE_2(3)$"])
    np.savetxt("figures/intro_T.txt", T[:, -1, :3, 4].numpy(), header="x y z", comments='')
    plt.show()


if __name__ == '__main__':
    ### Parameters ###
    i_max = 1200 # number of random points
    k_max = 301 # number of compounded poses
    sigma = 3 # plot option (factor for ellipse size)
    m_max = 100 # plot option (number of points for each ellipse)
    g = torch.Tensor([0, 0, 9.81]) # gravity vector
    dt = 0.01 # step time (s)

    # Define a PDF over transformations
    Upsilon = torch.eye(5)
    Upsilon[:3, 3] = -g*dt
    Upsilon[:3, 4] = Upsilon[:3, 3]*dt/2
    # Define right perturbation noise
    cholQ = Upsilon.new_zeros(9, 9)
    cholQ[:3, :3] = 1e-2*torch.eye(3)
    cholQ[3:6, 3:6] = 5e-3*torch.eye(3)
    cholQ[6:9, 3:6] = cholQ[3:6, 3:6]*dt/2
    Q = cholQ.mm(cholQ.t())
    # initial state
    T0 = torch.eye(5)
    T0[:3, 3] = torch.Tensor([5, 0, 0])
    # initial right perturbation uncertainty
    P0 = torch.zeros(9, 9)
    main(i_max, k_max, T0, P0, Upsilon, Q, cholQ, dt, g, sigma, m_max)

