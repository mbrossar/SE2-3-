import torch
from utils import *
from lie_group_utils import SO3, SE3_2
from preintegration_utils import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
torch.set_default_dtype(torch.float64)


def propagate(T0, P, Upsilon, Q, method, dt, g):
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

    Pprop = (Pprop + Pprop.t())/2 # symmetric
    return T, Pprop

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
        np.savetxt("figures/figure2"+color+str(n)+".txt", clines[:, :2].numpy(), comments="", header="x y")

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
        
        np.savetxt("figures/figure2"+color+str(n)+".txt", clines[:, :2].numpy(), comments="", header="x y")

def main(i_max, k_max, T0, P0, Upsilon, Q, cholQ, dt, g, sigma, m_max):
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
    P_est_b = torch.zeros(k_max, 9, 9) # SO(3) x R^6 covariance
    T_est[0] = T0
    P_est1[0] = P0.clone()
    P_est2[0] = P0.clone()
    P_est_b[0] = P0.clone()
    for k in range(1, k_max):
        # Second-order method
        T_est[k], P_est1[k] = propagate(T_est[k-1], P_est1[k-1], Upsilon, Q, 0, dt, g)
        # Fourth-order method
        _, P_est2[k] = propagate(T_est[k-1], P_est2[k-1], Upsilon, Q, 1, dt, g)
        # baseline method
        _, P_est_b[k] = propagate(T_est[k-1], P_est_b[k-1], Upsilon, Q, 2, dt, g)

    # Plot the random samples' trajectory lines
    for i in range(i_max):
        plt.plot(T[i, :, 0, 4], T[i, :, 1, 4], color='yellow')

    v = (2*np.pi*torch.arange(m_max)/(m_max-1) -np.pi).unsqueeze(1)
    x = T[:, -1, :3, 4]  
    xmean = torch.mean(x, dim=0)
    vSigma = bouter(x - xmean, x - xmean).sum(dim=0)/(i_max-1)
    for i1, i2 in [(0, 1)]:
        plt.figure()
        # Plot blue dots for random samples
        plt.scatter(T[:, -1, i1, 4], T[:, -1, i2, 4], s=1, color='b')
        # Plot the mean of the samples
        plt.scatter(xmean[i1], xmean[i2], label='mean', color='orange')
        # Plot the covariance of the samples
        T_est2 = T_est.clone()
        T_est2[-1, :3, 4] = xmean
        plot_so3_helper(T_est2, vSigma, v, "blue", i1, i2)
        # Plot the propagated mean in SE(3) - projected onto i1, i2
        plt.scatter(T_est[-1, i1, 4], T_est[-1, i2, 4], label='estimation',
            color='black')
        # Plot the propagated covariance projected onto i1, i2
        plot_se23_helper(T_est, P_est1[-1], v, 'red', i1, i2)
        plot_so3_helper(T_est, P_est_b[-1, 6:9, 6:9], v, 'cyan', i1, i2)
        plot_se23_helper(T_est, P_est2[-1], v,  'green', i1, i2)
        plt.xlabel('x')
        plt.xlim(left=0)
        plt.ylabel('y')
    plt.show()
    
    np.savetxt("figures/figure2T.txt", T[:, -1, :2, 4].numpy(), comments="", header="x y")
    
    b = T.new_zeros(int(T.shape[1]/10)+1, T.shape[0])
    header = ""
    for j in range(int(T.shape[0]/2)):
        b[:, 2*j:2*(j+1)] = T[j, ::10, :2, 4]
        header += str(j) + "x " + str(j) + str("y ")
    np.savetxt("figures/figure2traj.txt", b.numpy(), comments="", header=header)
    print("mean Gaussian", xmean)
    print("mean SE_2(3)", T_est[-1, :2, 4])

if __name__ == '__main__':
    ### Parameters ###
    i_max = 1000 # number of random points
    k_max = 301 # number of compounded poses
    sigma = 3 # plot option (factor for ellipse size)
    m_max = 30 # plot option (number of points for each ellipse)
    g = torch.Tensor([0, 0, 9.81]) # gravity vector
    dt = 0.05 # step time (s)

    # Constant acceleration, noise on IMU
    # Define a PDF over transformations (mean and covariance)
    xibar = torch.Tensor([0, 0, 0, 1, 0, 0, 0, 0, 0]).cuda()*dt
    Upsilon = SE3_2.uexp(xibar).cpu()
    Upsilon[:3, 3] += -g*dt
    Upsilon[:3, 4] = Upsilon[:3, 3]*dt/2

    T0 = torch.eye(5)
    P0 = torch.zeros(9, 9)
    # Define right perturbation noise
    cholQ = torch.Tensor([0, 0, 0.03, 0, 0, 0, 0, 0, 0]).diag()
    Q = cholQ.mm(cholQ.t())
    main(i_max, k_max, T0, P0, Upsilon, Q, cholQ, dt, g, sigma, m_max)

