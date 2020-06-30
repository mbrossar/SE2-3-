import torch
from utils import *
from lie_group_utils import SO3, SE3_2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.linalg
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=8)

def save_bias_errors(results, alphas):
    base_name = "figures/bias"
    data = np.zeros((alphas.shape[0], 13))
    data[:, 0] = alphas.numpy()

    names = ["se32.txt","so3.txt"]
    header = "alpha deltaR_mean deltaR_med deltaR_minus deltaR_plus deltaV_mean deltaV_med deltaV_minus deltaV_plus deltaP_mean deltaP_med deltaP_minus deltaP_plus"

    for i in range(2):
        name = base_name + names[i]
        res = results[:, :, 3*i]
        data[:, 1] = res.mean(dim=1).numpy()
        data[:, 2] = res.median(dim=1)[0].numpy()
        for k in range(alphas.shape[0]):
            data[k, 3] = percentile(res[k], 33)
            data[k, 4] = percentile(res[k], 67)
        res = results[:, :, 3*i+1]
        data[:, 5] = res.mean(dim=1).numpy()
        data[:, 6] = res.median(dim=1)[0].numpy()
        for k in range(alphas.shape[0]):
            data[k, 7] = percentile(res[k], 33)
            data[k, 8] = percentile(res[k], 67)
        res = results[:, :, 3*i+2]
        data[:, 9] = res.mean(dim=1).numpy()
        data[:, 10] = res.median(dim=1)[0].numpy()
        for k in range(alphas.shape[0]):
            data[k, 11] = percentile(res[k], 33)
            data[k, 12] = percentile(res[k], 67)
        np.savetxt(name, data, header=header, comments="")

def correctPIM(DeltaRs, DeltaVs, DeltaPs, DUpsilonDb, biases, methode):
    """
    Correct preintegration measurement of SE_2(3) and SO(3)xR^6 distributions.
    """
    N = DeltaRs.shape[0]
    delta_pim = bmv(DUpsilonDb.expand(N, 9, 6), biases)

    if methode == 1:
        # Correct preintegration measurement of SE_2(3) distribution.
        Upsilon = torch.eye(5)
        Upsilon[:3, :3] = DeltaRs[0]
        Upsilon[:3, 3] = DeltaVs[0]
        Upsilon[:3, 4] = DeltaPs[0]

        Upsilon_cor = Upsilon.expand(N, 5, 5).bmm(SE3_2.exp(delta_pim.cuda()).cpu())
        DeltaRs_cor = Upsilon_cor[:, :3, :3]
        DeltaVs_cor = Upsilon_cor[:, :3, 3]
        DeltaPs_cor = Upsilon_cor[:, :3, 4]
    else:
        # Correct preintegration measurement of SO(3)xR^6 distribution.
        DeltaRs_cor = DeltaRs[0].expand(N, 3, 3).bmm(SO3.exp(delta_pim[:, :3].cuda()).cpu())
        DeltaVs_cor = DeltaVs[0].expand(N, 3) + delta_pim[:, 3:6]
        DeltaPs_cor = DeltaPs[0].expand(N, 3) + delta_pim[:, 6:9]

    return DeltaRs_cor, DeltaVs_cor, DeltaPs_cor

def computePIM(omegas_m, accs_m, biases, dt):
    Pim = torch.eye(5).repeat(biases.shape[0], 1, 1)
    Upsilon = torch.eye(5).repeat(biases.shape[0], 1, 1)

    for k in range(omegas_m.shape[0]):
        omega = omegas_m[k] + biases[:, :3]
        acc = accs_m[k] + biases[:, 3:6]
        Upsilon[:, :3, :3] = SO3.exp((omega*dt).cuda()).cpu()
        Upsilon[:, :3, 3] = acc*dt
        Upsilon[:, :3, 4] = 1/2*acc*(dt**2)
        Pim = Pim.bmm(Upsilon)

    DeltaRs = Pim[:, :3, :3]
    DeltaVs = Pim[:, :3, 3]
    DeltaPs = Pim[:, :3, 4]
    return DeltaRs, DeltaVs, DeltaPs

def computeJacobian(omegas_m, accs_m, biases, dt):
    """
    Compute first-order Jacobian for bias update.
    Note: same for each distribution.
    """
    J = torch.zeros(9, 6)
    G = torch.zeros(9, 6)
    G[:3, :3] = dt*torch.eye(3)
    G[3:6, 3:6] = dt*torch.eye(3)
    G[6:9, 3:6] = G[3:6, 3:6]*dt/2
    Upsilon = torch.eye(5)
    for k in range(accs_m.shape[0]):
        omega = omegas_m[k]
        acc = accs_m[k]
        Upsilon[:3, :3] = SO3.uexp((omega*dt)).cpu()
        Upsilon[:3, 3] = acc*dt
        Upsilon[:3, 4] = 1/2*acc*(dt**2)
        J = SE3_2.uAd(Upsilon.inverse()).mm(J) + G
    return J

### Parameters ###
i_max = 100000 # number of random points
k_max = 50 # number of preintegration
g = torch.Tensor([0, 0, 9.81]) # gravity vector
dt = 0.1 # step time (s)
alphas = torch.arange(20)*0.02

# Load inputs
data = pload('figures/10.p')
omegas = data[:, 10:13]
accs = data[:, 13:16]
M = int(data.shape[0]/k_max) # number of preintegrated measurements

# random biases of norm 1
biases = torch.randn(i_max, 6)
biases[:, :3] /= 10
biases = biases/biases.norm(dim=1, keepdim=True)
biases[0] *= 0 # reference at index 0

results = torch.zeros(alphas.shape[0], M, 6)

for m in range(M):
    print(m/M)
    # compute true preintegration factor with/without biases
    omegas_m = omegas[m*k_max: (m+1)*k_max]
    accs_m = accs[m*k_max: (m+1)*k_max]
    DeltaRs, DeltaVs, DeltaPs = computePIM(omegas_m, accs_m, biases, dt)

    # compute first-order Jacobian
    DUpsilonDb = computeJacobian(omegas_m, accs_m, biases, dt)
    for k in range(alphas.shape[0]):
        a = alphas[k]
        # evaluate corrected preintegration factor
        DeltaRs_SO3, DeltaVs_SO3, DeltaPs_SO3 = correctPIM(DeltaRs, DeltaVs, DeltaPs, DUpsilonDb, a*biases, 2)
        DeltaRs_SE23, DeltaVs_SE23, DeltaPs_SE23 = correctPIM(DeltaRs, DeltaVs, DeltaPs, DUpsilonDb, a*biases, 1)
        DeltaRs, DeltaVs, DeltaPs = computePIM(omegas_m, accs_m, a*biases, dt)
        # evaluate error
        results[k, m, 0] = SO3.log(bmtm(DeltaRs[1:], DeltaRs_SE23[1:]).cuda()).cpu().norm()/(i_max-1)
        results[k, m, 1] = (DeltaVs[1:] - DeltaVs_SE23[1:]).norm()/(i_max-1)
        results[k, m, 2] = (DeltaPs[1:] - DeltaPs_SE23[1:]).norm()/(i_max-1)
        results[k, m, 3] = SO3.log(bmtm(DeltaRs[1:], DeltaRs_SO3[1:]).cuda()).cpu().norm()/(i_max-1)
        results[k, m, 4] = (DeltaVs[1:] - DeltaVs_SO3[1:]).norm()/(i_max-1)
        results[k, m, 5] = (DeltaPs[1:] - DeltaPs_SO3[1:]).norm()/(i_max-1)

for j in range(3):
    a = results[:, :, j].mean(dim=1)
    b = results[:, :, 3+j].mean(dim=1)
    plt.figure(),plt.plot(a),plt.plot(b)
plt.show()
# save
save_bias_errors(results, alphas)


