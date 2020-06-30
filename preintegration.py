import torch
from utils import *
from lie_group_utils import SO3, SE3_2
from preintegration_utils import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.linalg
from torch.distributions.multivariate_normal import MultivariateNormal
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=8)



def plot_se23_helper(T_est, P_est, color, i1, i2, i_max):
    P_est_chol = torch.cholesky(P_est + torch.eye(9)*1e-16)
    r = bmv(P_est_chol.expand(i_max, 9, 9), torch.randn(i_max, 9))
    Ttemp = T_est[-1].expand(i_max, 5, 5).bmm(SE3_2.exp(r.cuda()).cpu())
    p_est = Ttemp[:, :3, 4]
    plt.scatter(p_est[:, i1], p_est[:, i2], color=color, s=3)

def plot_so3_helper(T_est, P_est, color, i1, i2, i_max):
    r = torch.randn(i_max, 9)
    P_est_chol = torch.cholesky(P_est + torch.eye(9)*1e-16)
    p_est = bmv(P_est_chol.expand(i_max, 9, 9), r)[:, 6:9] + T_est[-1, :3, 4]
    plt.scatter(p_est[:, i1], p_est[:, i2], color=color, s=2, alpha=0.5)

def compound(T0, P, Upsilon, Q, method, dt, g, cholQ=0):
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
        Pprop = axat(F, P) + axat(G, Q[:6, :6])

    elif method == 4:
        # Monte Carlo method
        n_tot_samples = 100000
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

    elif method == 5:

        # Monte Carlo method
        n_tot_samples = 100000
        nsamples = 50000
        N = int(n_tot_samples/nsamples)+1

        tmp = torch.cholesky(P + 1e-20*torch.eye(9))
        cholP = tmp.cuda().expand(nsamples, 9, 9)
        cholQ = cholQ.cuda().expand(nsamples, 9, 9)

        Pprop = torch.zeros(9, 9)

        Gamma = Gamma.cuda().expand(nsamples, 5, 5)
        Upsilon = Upsilon.cuda().expand(nsamples, 5, 5)
        T0 = T0.cuda().expand(nsamples, 5, 5)
        T_hat = T.cuda().expand(nsamples, 5, 5)
        for i in range(N):
            xi0 = bmv(cholP, torch.randn(nsamples, 9).cuda())
            w = bmv(cholQ, torch.randn(nsamples, 9).cuda())
            T0_i = SE3_2.boxplus(T0, xi0)
            Phi = f_flux(T0_i, dt)
            Upsilon_i = Upsilon.bmm(SE3_2.exp(w))
            T_i = Gamma.bmm(Phi).bmm(Upsilon_i)
            xi = SE3_2.boxminus(T_i, T_hat)
            xi_mean = xi.mean(dim=0)
            Pprop += bouter(xi-xi_mean, xi-xi_mean).sum(dim=0).cpu()

        Pprop = Pprop / (N*nsamples+1)

    Pprop = (Pprop + Pprop.t())/2
    return T, Pprop

def bdot(a, b):
    return torch.einsum('bi, bi -> b', a, b)

def compute_nees(P, xi):
    Pinv = (P[-1] + 1e-16*torch.eye(9)).inverse().expand(xi.shape[0], 9, 9)
    return bdot(xi, bmv(Pinv, xi)).mean()/9

def print_kl_div(T, i_max, T_est, P_est, P_est_b, P_est_mc, P_est_b_mc):
    results = torch.zeros(8)
    ## KL divergence w.r.t. Monte-Carlo sample ##
    loss_class = torch.nn.KLDivLoss()
    target_prob = 1/i_max*torch.ones(i_max)

    # Methods on SE_2(3)
    chi_diff = SE3_2.uinv(T_est[-1]).expand(i_max, 5, 5).bmm(T[:, -1])
    xi = SE3_2.log(chi_diff.cuda()).cpu()
    s_dist = MultivariateNormal(torch.zeros(9),
        covariance_matrix=P_est[-1] + 1e-16*torch.eye(9))
    mc_dist = MultivariateNormal(torch.zeros(9), # Monte-Carlo
        covariance_matrix=P_est_mc[-1] + 1e-16*torch.eye(9))
    input_s_log_prob = s_dist.log_prob(xi)
    input_mc_log_prob = mc_dist.log_prob(xi)

    s_loss = loss_class(input_s_log_prob, target_prob)
    mc_loss = loss_class(input_mc_log_prob, target_prob)

    print('KL divergence w.r.t. Monte-Carlo samples')
    print('$SE_2(3)$: {:.8f}'.format(s_loss))
    print('$SE_2(3)$ Monte-Carlo: {:.8f}'.format(mc_loss))
    results[0] = s_loss
    results[1] = mc_loss

    s_nees = compute_nees(P_est, xi)  
    mc_nees = compute_nees(P_est_mc, xi)  

    print('NEES w.r.t. Monte-Carlo samples')
    print('$SE_2(3)$: {:.8f}'.format(s_nees))
    print('$SE_2(3)$ Monte-Carlo: {:.8f}'.format(mc_nees))
    results[2] = s_nees
    results[3] = mc_nees

    # Methods on SO(3)
    xi = SE3_2.boxminus(T_est[-1].expand(i_max, 5, 5).cuda(), T[:, -1].cuda()).cpu()

    s_dist = MultivariateNormal(torch.zeros(9),
        covariance_matrix=P_est_b[-1] + 1e-16*torch.eye(9))
    mc_dist = MultivariateNormal(torch.zeros(9), # Monte-Carlo
        covariance_matrix=P_est_b_mc[-1] + 1e-16*torch.eye(9))
    input_s_log_prob = s_dist.log_prob(xi)
    input_mc_log_prob = mc_dist.log_prob(xi)
    s_loss = loss_class(input_s_log_prob, target_prob)
    mc_loss = loss_class(input_mc_log_prob, target_prob)
    print('KL divergence w.r.t. Monte-Carlo samples')
    print('$SO(3)$: {:.8f}'.format(s_loss))
    print('$SO(3)$ Monte-Carlo: {:.8f}'.format(mc_loss))
    results[4] = s_loss
    results[5] = mc_loss

    s_nees = compute_nees(P_est_b, xi)  
    mc_nees = compute_nees(P_est_b_mc, xi)  

    print('NEES w.r.t. Monte-Carlo samples')
    print('$SO(3)$: {:.8f}'.format(s_nees))
    print('$SO(3)$ Monte-Carlo : {:.8f}'.format(mc_nees))
    results[6] = s_nees
    results[7] = mc_nees

    print()
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
    P_est = torch.zeros(k_max, 9, 9) # covariance
    P_est_b = torch.zeros(k_max, 9, 9) # SO(3) x R^6 covariance
    P_est_mc = torch.zeros(k_max, 9, 9) # Monte-Carlo covariance on SE_2(3)
    P_est_b_mc = torch.zeros(k_max, 9, 9) # Monte-Carlo covariance on SO(3) x R^6
    T_est[0] = T0
    for k in range(1, k_max):
        # Second-order method
        T_est[k], P_est[k] = compound(T_est[k-1], P_est[k-1], Upsilons[k], Q, 1, dt, g)
        # baseline method
        _, P_est_b[k] = compound(T_est[k-1], P_est_b[k-1], Upsilons[k], Q, 3, dt, g)
        # Monte-Carlo method
        _, P_est_mc[k] = compound(T_est[k-1], P_est_mc[k-1], Upsilons[k], Q, 4, dt, g, cholQ)
        _, P_est_b_mc[k] = compound(T_est[k-1], P_est_b_mc[k-1], Upsilons[k], Q, 5, dt, g, cholQ)

    results = print_kl_div(T, i_max, T_est, P_est, P_est_b, P_est_mc, P_est_b_mc)
    return results

def save_kl_div_and_cov(results, alphas):
    base_name = "figures/preintegration"
    data = np.zeros((alphas.shape[0], 9))
    data[:, 0] = alphas.numpy()

    names = ["se32.txt","se32mc.txt","so3.txt","so3_mc.txt"]
    header = "alpha kldiv_mean kldiv_med kldiv_minus kldiv_plus nees_mean nees_med nees_minus nees_plus"

    for i in range(2):
        name = base_name + names[i]
        res = results[:, :, i]
        data[:, 1] = res.mean(dim=1).numpy()
        data[:, 2] = res.median(dim=1)[0].numpy()
        for k in range(alphas.shape[0]):
            data[k, 3] = percentile(res[k], 33)
            data[k, 4] = percentile(res[k], 67)
        res = results[:, :, i+2]
        data[:, 5] = res.mean(dim=1).numpy()
        data[:, 6] = res.median(dim=1)[0].numpy()
        for k in range(alphas.shape[0]):
            data[k, 7] = percentile(res[k], 33)
            data[k, 8] = percentile(res[k], 67)
        # plt.plot(data[:, [2,3,4]]),plt.figure()
        # plt.plot(data[:, [6,7,8]]),plt.show()
        np.savetxt(name, data, header=header, comments="")

    for i in range(2, 4):
        name = base_name + names[i]
        res = results[:, :, i+2]
        data[:, 1] = res.mean(dim=1).numpy()
        data[:, 2] = res.median(dim=1)[0].numpy()
        for k in range(alphas.shape[0]):
            data[k, 3] = percentile(res[k], 33)
            data[k, 4] = percentile(res[k], 67)
        res = results[:, :, i+4]
        data[:, 5] = res.mean(dim=1).numpy()
        data[:, 6] = res.median(dim=1)[0].numpy()
        for k in range(alphas.shape[0]):
            data[k, 7] = percentile(res[k], 33)
            data[k, 8] = percentile(res[k], 67)
        plt.plot(data[:, [2,3,4]]),plt.figure()
        plt.plot(data[:, [6,7,8]]),plt.show()
        np.savetxt(name, data, header=header, comments="")



### Parameters ###
i_max = 100000 # number of random points
k_max = 50 # number of preintegration
sigma_omega = 0.01 # standard deviation of gyro
sigma_acc = 0.01 # standard deviation of acc
g = torch.Tensor([0, 0, 9.81]) # gravity vector
dt = 0.1 # step time (s)
alphas = torch.Tensor([0.1, 0.5, 1, 2, 3, 5, 8, 10, 20])

# Load inputs
data = pload('figures/10.p')
Rots = SO3.from_quaternion(data[:, :4])
vs = data[:, 4:7]
ps = data[:, 7:10]
omegas = data[:, 10:13]
accs = data[:, 13:16]
M = int(data.shape[0]/k_max) # number of preintegrated measurements
results = torch.zeros(alphas.shape[0], M, 8)

# Define right perturbation noise
cholQ = torch.zeros((9, 9))
cholQ[:3, :3] = sigma_omega * np.sqrt(dt) * torch.eye(3)
cholQ[3:6, 3:6] = sigma_acc * np.sqrt(dt) * torch.eye(3)
cholQ[3:6, 6:9] = cholQ[3:6, 3:6]*dt/2
Q = cholQ.mm(cholQ.t())
T0 = torch.eye(5)
Upsilons = torch.eye(5).repeat(k_max, 1, 1)

for d in range(alphas.shape[0]):
    alpha = alphas[d]
    aQ =  (alpha**2)*Q
    acholQ = alpha*cholQ
    for m in range(M):
        print("alpha:", alpha.item(), "iteration:", m)
        T0[:3, :3] = Rots[m*k_max]
        T0[:3, 3] = vs[m*k_max]
        T0[:3, 4] = ps[m*k_max]
        omegas_m = omegas[m*k_max: (m+1)*k_max]
        accs_m = accs[m*k_max: (m+1)*k_max]
        Upsilons[:, :3, :3] = SO3.exp((omegas_m*dt).cuda()).cpu()
        Upsilons[:, :3, 3] = accs_m*dt
        Upsilons[:, :3, 4] = 1/2*accs_m*(dt**2)

        results[d, m] = main(i_max, k_max, T0, Upsilons, aQ, acholQ, dt, g)
pdump(results, "figures/preintegration.p")
results = pload("figures/preintegration.p")
save_kl_div_and_cov(results, alphas)


