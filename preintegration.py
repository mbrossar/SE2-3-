import torch
from utils import *
from lie_group_utils import SO3, SE3_2
from preintegration_utils import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
torch.set_default_dtype(torch.float64)


def compound(T0, Sigma, Upsilon, Q, method, dt, g, cholQ=0):
    Gamma = f_Gamma(g, dt)
    Phi = f_flux(T0, dt)
    # compound the mean
    T = Gamma.mm(Phi).mm(Upsilon)

    # Jacobian for propagating prior along time
    F = torch.eye(9)
    F[6:9, 3:6] = torch.eye(3)*dt

    # compute Adjoint of right transformation mean
    AdUps = SE3_2.uAd(SE3_2.uinv(Upsilon))
    Sigma_tmp = axat(AdUps.mm(F), Sigma)
    # compound the covariances based on the second-order method
    Sigma_prop = Sigma_tmp + Q

    if method == 3:
        # baseline SO(3) x R^6
        wedge_acc = SO3.uwedge(Upsilon[:3, 3]) # already multiplied by dt
        F = torch.eye(9)
        F[3:6, :3] = T0[:3, :3].t()
        F[3:6, :3] = -T0[:3, :3].mm(wedge_acc)
        F[6:9, :3] = F[3:6, :3]*dt/2
        F[6:9, 3:6] = dt*torch.eye(3)

        G = torch.zeros(9, 6)
        G[:3, :3] = T0[:3, :3].t()
        G[3:6, 3:6] = T0[:3, :3]
        G[6:9, 3:6] = 1/2*T0[:3, :3]*dt
        Sigma_prop = axat(F, Sigma) + axat(G, Q[:6, :6])

    elif method == 4:
        # Monte Carlo method
        n_tot_samples = 100000
        nsamples = 50000
        N = int(n_tot_samples/nsamples)+1

        tmp = torch.cholesky(Sigma_prop + 1e-16*torch.eye(9))
        cholP = tmp.cuda().expand(nsamples, 9, 9)
        cholQ = cholQ.cuda().expand(nsamples, 9, 9)

        Sigma_prop = torch.zeros(9, 9)

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
            Sigma_prop += bouter(xi-xi_mean, xi-xi_mean).sum(dim=0).cpu()

        Sigma_prop = Sigma_prop / (N*nsamples+1)


        Sigma_prop = Sigma_prop / (N*nsamples+1)

    Sigma_prop = (Sigma_prop + Sigma_prop.t())/2
    return T, Sigma_prop

def bdot(a, b):
    return torch.einsum('bi, bi -> b', a, b)

def compute_nees(P, xi):
    Pinv = (P[-1] + 1e-16*torch.eye(9)).inverse().expand(xi.shape[0], 9, 9)
    return bdot(xi, bmv(Pinv, xi)).mean()/9

def compute_results(T, i_max, T_est, Sigma_est, SigmaSO3, Sigma_est_mc):
    results = torch.zeros(3)

    # Methods on SE_2(3)
    chi_diff = SE3_2.uinv(T_est[-1]).expand(i_max, 5, 5).bmm(T[:, -1])
    xi = SE3_2.log(chi_diff.cuda()).cpu()
    s_nees = compute_nees(Sigma_est, xi)  
    mc_nees = compute_nees(Sigma_est_mc, xi)  
    results[0] = s_nees
    results[1] = mc_nees

    # Method on SO(3)
    xi = SE3_2.boxminus(T_est[-1].expand(i_max, 5, 5).cuda(), T[:, -1].cuda()).cpu()
    s_nees = compute_nees(SigmaSO3, xi)
    results[2] = s_nees
    return results

def main(i_max, k_max, T0, Upsilons, Q, cholQ, dt, g):
    # Generate some random samples
    # NOTE: initial covariance is zero
    T = torch.zeros(i_max, k_max, 5, 5).cuda()
    T[:, 0] = T0.cuda().repeat(i_max, 1, 1)
    Gamma = f_Gamma(g, dt).cuda().expand(i_max, 5, 5)
    tmp = cholQ.cuda().expand(i_max, 9, 9)
    for k in range(1, k_max):
        T_k = SE3_2.exp(bmv(tmp, torch.randn(i_max, 9).cuda()))
        Phi = f_flux(T[:, k-1], dt)
        tmp2 = Upsilons[k].cuda().expand(i_max, 5, 5)
        T[:, k] = Gamma.bmm(Phi).bmm(T_k).bmm(tmp2)
    T = T.cpu()

    # Propagate the uncertainty using second- and fourth-order methods
    T_est = torch.zeros(k_max, 5, 5)
    Sigma_est = torch.zeros(k_max, 9, 9) # covariance
    SigmaSO3 = torch.zeros(k_max, 9, 9) # SO(3) x R^6 covariance
    Sigma_est_mc = torch.zeros(k_max, 9, 9) # Monte-Carlo covariance on SE_2(3)

    T_est[0] = T0
    for k in range(1, k_max):
        # Second-order method
        T_est[k], Sigma_est[k] = compound(T_est[k-1], Sigma_est[k-1], Upsilons[k], Q, 1, dt, g)
        # baseline method
        _, SigmaSO3[k] = compound(T_est[k-1], SigmaSO3[k-1], Upsilons[k], Q, 3, dt, g)
        # Monte-Carlo method
        _, Sigma_est_mc[k] = compound(T_est[k-1], Sigma_est_mc[k-1], Upsilons[k], Q, 4, dt, g, cholQ)


    results = compute_results(T, i_max, T_est, Sigma_est, SigmaSO3, Sigma_est_mc)
    return results

def save_results(vs, results, k_maxs, alpha):
    """Save results to paper"""
    base_name = "figures/preintegration"
    data = np.zeros((len(k_maxs), 9))
    data[:, 0] = 0.1*np.array(k_maxs)

    names = ["se32_","se32mc_","so3_"]
    for i in range(3):
        names[i] = names[i] + str(alpha) + ".txt"
    header = "dt nees_mean nees_med nees_minus nees_plus"

    for i in range(3):
        name = base_name + names[i]
        for k in range(data.shape[0]):
            M = int(vs.shape[0]/k_maxs[k])
            res = results[k, :M, i]
            data[k, 1] = res.mean(dim=0).numpy()
            data[k, 2] = res.median(dim=0)[0].numpy()
            data[k, 3] = percentile(res, 33)
            data[k, 4] = percentile(res, 67)
        np.savetxt(name, data, header=header, comments="")
        
def plot_results(vs, results, k_maxs, alpha):
    



### Parameters ###
i_max = 100000 # number of random points
k_maxs = [10, 40, 70, 100, 130, 160, 190, 210, 240, 270, 300] # number of preintegration
sigma_omega = 0.0007 * np.sqrt(10)  # standard deviation of gyro
sigma_acc = 0.0019 * np.sqrt(10) # standard deviation of acc
g = torch.Tensor([0, 0, 9.81]) # gravity vector
dt = 0.1 # step time (s)
alphas = torch.Tensor([0.1, 1, 10])

# Load inputs
data = pload('figures/10.p')
Rots = SO3.from_quaternion(data[:, :4])
vs = data[:, 4:7]
ps = data[:, 7:10]

omegas = data[:, 10:13]
accs = data[:, 13:16]

M_max = int(data.shape[0]/k_maxs[0])
results = torch.zeros(len(k_maxs), M_max, 4)

# Define right perturbation noise
cholQ = torch.zeros((9, 9))
cholQ[:3, :3] = sigma_omega * np.sqrt(dt) * torch.eye(3)
cholQ[3:6, 3:6] = sigma_acc * np.sqrt(dt) * torch.eye(3)
cholQ[3:6, 6:9] = cholQ[3:6, 3:6]*dt/2
Q = cholQ.mm(cholQ.t())
T0 = torch.eye(5)

for d in range(alphas.shape[0]):
    alpha = alphas[d]
    aQ =  (alpha**2)*Q
    acholQ = alpha*cholQ
    for k in range(len(k_maxs)):
        k_max = k_maxs[k]
        M = int(data.shape[0]/k_max) # number of preintegrated measurements
        Upsilons = torch.eye(5).repeat(k_max, 1, 1)
        for m in range(M):
            print("alpha:", alpha.item(),"k_max:", k_max, "iteration:", m)
            T0[:3, :3] = Rots[m*k_max]
            T0[:3, 3] = vs[m*k_max]
            T0[:3, 4] = ps[m*k_max]
            omegas_m = omegas[m*k_max: (m+1)*k_max]
            accs_m = accs[m*k_max: (m+1)*k_max]
            Upsilons[:, :3, :3] = SO3.exp((omegas_m*dt).cuda()).cpu()
            Upsilons[:, :3, 3] = accs_m*dt
            Upsilons[:, :3, 4] = 1/2*accs_m*(dt**2)
            results[k, m] = main(i_max, k_max, T0, Upsilons, aQ, acholQ, dt, g)

    pdump(results, "figures/preintegration_" + str(alpha) + ".p")


for d in range(alphas.shape[0]):        
    results = pload("figures/preintegration_" + str(alpha) + ".p")
    save_results(vs, results, k_maxs, alpha)
    plot_results(vs, results, k_maxs, alpha)

