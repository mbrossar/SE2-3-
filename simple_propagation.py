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
import numpy as np
import scipy.linalg
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

def propagate(T0, Sigma, Upsilon, Q, method, dt, g):
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

    elif method == 2:
        # SO(3) x R^6
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
        Sigma_prop = axat(F, Sigma) + axat(G, Q[:6, :6]/(dt**2))

    Sigma_prop = (Sigma_prop + Sigma_prop.t())/2 # symmetric
    return T, Sigma_prop

def plot_se23_helper(T_est, P_est, v, color, i1, i2):
    """
    Draw ellipse based on the 3 more important directions of the covariance
    """
    D, V = torch.eig(P_est, eigenvectors=True)
    Y, I = torch.sort(D[:, 0], descending=True)
    a = sigma*D[I[0], 0].sqrt() * V[:, I[0]]
    b = sigma*D[I[1], 0].sqrt() * V[:, I[1]]
    c = sigma*D[I[2], 0].sqrt() * V[:, I[2]]
    for n in range(3):
        if n == 0:
            xi = a*v.sin() + b*v.cos()
        elif n == 1:
            xi = b*v.sin() + c*v.cos()
        elif n == 2:
            xi = a*v.sin() + c*v.cos()
        Ttemp = T_est[-1].expand(m_max, 5, 5).bmm(SE3_2.exp(xi.cuda()).cpu())
        clines = Ttemp[:, :3, 4]
        plt.plot(clines[:, i1], clines[:, i2], color=color)
        # np.savetxt("figures/figure2"+color+str(n)+".txt", clines[:, :2].numpy(), comments="", header="x y")

def plot_so3_helper(T_est, P_est, v, color, i1, i2):
    """
    Draw ellipse based on the 3 more important directions of the covariance
    """
    D, V = torch.eig(P_est, eigenvectors=True)
    Y, I = torch.sort(D[:, 0], descending=True)
    a = sigma*D[I[0], 0].sqrt() * V[:, I[0]]
    b = sigma*D[I[1], 0].sqrt() * V[:, I[1]]
    c = sigma*D[I[2], 0].sqrt() * V[:, I[2]]
    for n in range(3):
        if n == 0:
            p = a*v.sin() + b*v.cos()
        elif n == 1:
            p = b*v.sin() + c*v.cos()
        elif n == 2:
            p = a*v.sin() + c*v.cos()
        clines = T_est[-1, :3, 4].expand(m_max, 3) + p
        plt.plot(clines[:, i1], clines[:, i2], color=color)
        # np.savetxt("figures/figure2"+color+str(n)+".txt", clines[:, :2].numpy(), comments="", header="x y")

def main(i_max, k_max, T0, Sigma0, Upsilon, Q, cholQ, dt, g, sigma, m_max):
    # Generate some random samples
    T = torch.zeros(i_max, k_max, 5, 5).cuda()
    T[:, 0] = T0.cuda().repeat(i_max, 1, 1)
    tmp = Sigma0.sqrt().cuda().expand(i_max, 9, 9) # Pxi assumed diagonal!
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
    SigmaSO3 = torch.zeros(k_max, 9, 9) # SO(3) x R^6 covariance
    T_est[0] = T0
    Sigma2th[0] = Sigma0.clone()
    Sigma4th[0] = Sigma0.clone()
    SigmaSO3[0] = Sigma0.clone()
    for k in range(1, k_max):
        # Second-order method
        T_est[k], Sigma2th[k] = propagate(T_est[k-1], Sigma2th[k-1], Upsilon, Q, 0, dt, g)
        # Fourth-order method
        _, Sigma4th[k] = propagate(T_est[k-1], Sigma4th[k-1], Upsilon, Q, 1, dt, g)
        # baseline method
        _, SigmaSO3[k] = propagate(T_est[k-1], SigmaSO3[k-1], Upsilon, Q, 2, dt, g)

    ## Numerical check of paper formulas
    # Sigma_K = Sigma2th[-1]
    # sigma = Q[2, 2].sqrt()
    # K = k_max-1
    # a = 1
    # Deltat = 0.05
    
    # Sigma_phiphi = K * sigma * sigma
    # print(Sigma_phiphi, Sigma_K[2, 2])

    # Sigma_phiv = -(K-1)/2 * a * Deltat * Sigma_phiphi
    # print(Sigma_phiv, Sigma_K[2, 4])

    # Sigma_phip = (K-1)*(2*K-1)/12 * a * (Deltat**2) * Sigma_phiphi
    # print(Sigma_phip, Sigma_K[2, 7])

    # Sigma_vv = (K-1)*(2*K-1)/6 * ((a * Deltat)**2) * Sigma_phiphi
    # print(Sigma_vv, Sigma_K[4, 4])

    # Sigma_vp = (K-1)**2 * (K)**2 * 1/(8*K) * ((a**2) * (Deltat**3)) * Sigma_phiphi
    # print(Sigma_vp, Sigma_K[4, 7])
    
    # Sigma_pp = (K-1)*(2*K-1)*(3*(K-1)**2 + 3*K -4)/120 * ((a**2) * (Deltat**4)) * Sigma_phiphi
    # print(Sigma_pp, Sigma_K[7, 7])

    # Plot the random samples' trajectory lines
    for i in range(i_max):
        plt.plot(T[i, :, 0, 4], T[i, :, 1, 4], color='gray', alpha=0.1)

    v = (2*np.pi*torch.arange(m_max)/(m_max-1) -np.pi).unsqueeze(1)
    x = T[:, -1, :3, 4]
    xmean = torch.mean(x, dim=0)
    vSigma = bouter(x - xmean, x - xmean).sum(dim=0)/(i_max-1)

    # Plot blue dots for random samples
    plt.scatter(T[:, -1, 0, 4], T[:, -1, 1, 4], s=2, color='black')
    # Plot the mean of the samples
    plt.scatter(xmean[0], xmean[1], label='mean', color='orange')
    # Plot the covariance of the samples
    T_est2 = T_est.clone()
    T_est2[-1, :3, 4] = xmean
    plot_so3_helper(T_est2, vSigma, v, "orange", 0, 1)
    plt.scatter(T_est[-1, 0, 4], T_est[-1, 1, 4], label='estimation',
        color='green')
    plot_se23_helper(T_est, Sigma2th[-1], v, 'green', 0, 1)
    plot_so3_helper(T_est, SigmaSO3[-1, 6:9, 6:9], v, 'red', 0, 1)
    plot_se23_helper(T_est, Sigma4th[-1], v,  'cyan', 0, 1)
    plt.xlabel('x')
    plt.xlim(left=0)
    plt.ylabel('y')
    plt.show()
    #np.savetxt("figures/figure2T.txt", T[:, -1, :2, 4].numpy(), comments="", header="x y")

    # b = T.new_zeros(int(T.shape[1]/10)+1, T.shape[0])
    # header = ""
    # for j in range(int(T.shape[0]/2)):
    #     b[:, 2*j:2*(j+1)] = T[j, ::10, :2, 4]
    #     header += str(j) + "x " + str(j) + str("y ")
    #np.savetxt("figures/figure2traj.txt", b.numpy(), comments="", header=header)

if __name__ == '__main__':
    ### Parameters ###
    i_max = 1000 # number of random points
    k_max = 301 # number of compounded poses
    sigma = 3 # plot option (factor for ellipse size)
    m_max = 100 # plot option (number of points for each ellipse)
    g = torch.Tensor([0, 0, 9.81]) # gravity vector
    dt = 0.05 # step time (s)
    a = 1 # acceleration (m/s^2)

    # Constant acceleration, noise on IMU
    # Define a PDF over transformations (mean and covariance)
    xibar = torch.Tensor([0, 0, 0, a, 0, 0, 0, 0, 0]).cuda()*dt
    Upsilon = SE3_2.uexp(xibar).cpu()
    Upsilon[:3, 3] += -g*dt
    Upsilon[:3, 4] = Upsilon[:3, 3]*dt/2

    T0 = torch.eye(5)
    Sigma0 = torch.zeros(9, 9)
    # Define right perturbation noise
    cholQ = torch.Tensor([0, 0, 0.03, 0, 0, 0, 0, 0, 0]).diag()
    Q = cholQ.mm(cholQ.t())
    main(i_max, k_max, T0, Sigma0, Upsilon, Q, cholQ, dt, g, sigma, m_max)

