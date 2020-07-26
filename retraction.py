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
import scipy.linalg
from preintegration_utils import *
torch.manual_seed(0)


def plot_se23_helper(T_est, P_est, color, i1, i2, i_max, path):
    P_est_chol = torch.cholesky(P_est)
    r = bmv(P_est_chol.expand(i_max, 9, 9), torch.randn(i_max, 9))
    r = r[r.norm(dim=1) < 8.1682]
    Ttemp = T_est.expand(r.shape[0], 5, 5).bmm(SE3_2.exp(r.cuda()).cpu())
    p_est = Ttemp[:, :3, 4]
    plt.scatter(p_est[:, i1], p_est[:, i2], color=color, s=3)
    if i1 == 0 and i2 == 1:
        np.savetxt(path, p_est.numpy(), header="x y z", comments='')

def plot_so3_helper(T_est, P_est, color, i1, i2, i_max, path):
    r = torch.randn(i_max, 9)
    r = r[r.norm(dim=1) < 4.1682][:, 6:9]
    P_est_chol = torch.cholesky(P_est[6:9, 6:9])
    p_est = bmv(P_est_chol.expand(r.shape[0], 3, 3), r) + T_est[:3, 4]
    plt.scatter(p_est[:, i1], p_est[:, i2], color=color, s=2, alpha=0.5)
    if i1 == 0 and i2 == 1:
        np.savetxt(path2, p_est.numpy(), header="x y z", comments='')

if __name__ == '__main__':
    i_max = 1000
    # Propagate the uncertainty methods
    T_est = torch.eye(5)
    T_est[:3, 4] = torch.Tensor([5, 0, 0])
    P_est = torch.diag(torch.Tensor([0.1, 0.1, 0.5, 1, 1, 1, 0.1, 0.1, 0.1]))
    P_est = axat(SE3_2.uAd(T_est.inverse()), P_est)
    SigmaSO3 = torch.eye(9)
    SigmaSO3[7, 7] = 5
    SigmaSO3[6, 6] = 0.5

    # saving paths
    path1 = "figures/retraction_b.txt"
    path2 = "figures/retraction_est.txt"


    labels = ['x (m)', 'y (m)', 'z (m)']
    for i1, i2 in [(0, 1)]:
        plt.figure()
        # Plot the covariance of the samples
        T_est[:3, 4] = torch.Tensor([-4, 0, 0])
        plot_so3_helper(T_est, SigmaSO3, "red", i1, i2, i_max, path1)
        plt.scatter(T_est[i1, 4], T_est[i2, 4], color='blue', s=20)
        # Plot the propagated covariance projected onto i1, i2
        T_est[:3, 4] = torch.Tensor([5, 0, 0])
        plot_se23_helper(T_est, P_est, 'green', i1, i2, i_max, path2)
        plt.scatter(T_est[i1, 4], T_est[i2, 4], color='blue', s=20)
        plt.xlabel(labels[i1])
        plt.ylabel(labels[i2])
        plt.legend([r"$SO(3) \times \mathbb{R}^6$", "mean", "$SE_2(3)$"])
    plt.show()

