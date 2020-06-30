import torch
from utils import *
from lie_group_utils import SO3, SE3_2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.linalg
from torch.distributions.multivariate_normal import MultivariateNormal
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=4)
from github.preintegration_utils import f_flux


def f_Gamma(g, dt, omega_coriolis):
    """Compute Gamma preintegration with Coriolis forces"""
    vec = torch.cat((omega_coriolis, g))*dt
    tmp = SE3.uexp(vec.cuda()).cpu()
    Omega = SO3.uwedge(omega_coriolis)
    Omega2 = Omega.mm(Omega)
    ang = omega_coriolis.norm()
    if ang == 0:
        mat = (dt**2)/2*torch.eye(3)
    else:
        mat = (dt**2)/2*torch.eye(3) + (dt*ang-(dt*ang).sin())*Omega/(ang**3) +\
            ((dt*ang).cos()-1 + ((dt*ang)**2)/2)*Omega2/(ang**4)
    Gamma = torch.eye(5)
    Gamma[:3, :3] = tmp[:3, :3]
    Gamma[:3, 3] = tmp[:3, 3]
    Gamma[:3, 4] = tmp[:3, :3].mm(mat).mv(g)
    return Gamma



def integrate(Upsilon, Rot, v, p, dt, N, Omega_coriolis, method, g):
    g = -g
    Gamma = f_Gamma(g, dt*N, Omega_coriolis)
    T0 = torch.eye(5)
    T0[:3, :3] = Rot
    T0[:3, 3] = v
    T0[:3, 4] = p
    Phi = f_flux(T0, dt*N)
    if method == 1:
        T = Gamma.mm(Phi.mm(Upsilon))
    else:
        T2 = Gamma.mm(Phi.mm(Upsilon))
        Gamma = f_Gamma(g, dt*N, 0*Omega_coriolis)
        T = Gamma.mm(Phi.mm(Upsilon))
        dv = - 2 * SO3.uwedge(Omega_coriolis).mv(v)* N * dt
        T[:3, :3] = T2[:3, :3]
        T[:3, 3] += dv
        T[:3, 4] += dv*dt/2
    Rot = T[:3, :3]
    v = T[:3, 3]
    p = T[:3, 4]
    return Rot, v, p

def integrate_imu(us, dt, N):
    Upsilon = torch.eye(5)
    Upsilon_i = torch.eye(5)
    Omega = SO3.exp(us[:N, :3].cuda()*dt).cpu()
    for i in range(N):
        Upsilon_i[:3, :3] = Omega[i]
        Upsilon_i[:3, 3] = us[i, 3:6]*dt
        Upsilon_i[:3, 4] = 1/2*us[i, 3:6]*(dt**2)
        Upsilon = f_flux(Upsilon, dt).mm(Upsilon_i)
    return Upsilon

def propagate_coriolis(us, Rot0, v0, p0, dt, Delta_t, N, N_tot, Omega_coriolis, method, g, Rots_gt):
    Rots = torch.zeros(N_tot, 3, 3)
    vs = torch.zeros(N_tot, 3)
    ps = torch.zeros(N_tot, 3)

    Rots[0] = Rot0
    vs[0] = v0
    ps[0] = p0
    Omega = SO3.exp(us[:, :3].cuda()*dt).cpu()
    Rots_gt = Rots_gt.cuda()

    for i in range(1, N_tot):
        print(i/N_tot)
        Upsilon = integrate_imu(us[i*N:], dt, N)
        Rots[i], vs[i], ps[i] = integrate(Upsilon, Rots[i-1], vs[i-1], ps[i-1], dt, N, Omega_coriolis, method, g)

    Rots = SO3.dnormalize(Rots.cuda()).cpu()
    return Rots, vs, ps

def compute_input(data, dt, g, Omega_coriolis):

    Rots = data["Rots"]
    ps = data["ps"]

    us = torch.zeros(ps.shape[0], 6)
    Omega = SO3.uwedge(Omega_coriolis).expand(us.shape[0], 3, 3)
    tmp = SO3.vee(bmtm(Rots, Omega).bmm(Rots))

    omegas = -tmp[1:] + 1/dt * SO3.log(bmtm(Rots[:-1], Rots[1:]).cuda()).cpu()
    vs = torch.zeros_like(ps)
    vs[1:] = (ps[1:]-ps[:-1])/dt
    g = g.expand(us.shape[0]-1, 3)
    tmp1 = 2 * bmv(Omega, vs)
    tmp2 =  bmv(Omega.bmm(Omega), vs)
    accs = bmtv(Rots[:-1], g + tmp1[:-1] + tmp2[:-1] + 1/dt*(vs[1:]-vs[:-1]))

    us[1:, :3] = omegas
    us[1:, 3:6] = accs
    data["vs"] = vs
    data["us"] = us
    pdump(data, "figures/coriolis.p")


# load ground-truth position and input
data = pload("figures/coriolis.p")
dt = 0.02 # (s)
Delta_t = 5
latitude = np.pi/180*48.7
earth_rate = 7.292115e-5 # rad/s
Omega_coriolis = earth_rate*torch.Tensor([0*np.cos(latitude), 0, -np.sin(latitude)]) # NED
g = torch.Tensor([0, 0, 9.81])
compute_input(data, dt, g, Omega_coriolis)
data = pload("figures/coriolis.p")
N = int(Delta_t/dt)
us = data["us"]
N_tot = int(us.shape[0]/N)
Rots = data["Rots"][:N_tot*N]
Rots[::N] = SO3.dnormalize(Rots[::N].cuda()).cpu()
vs = data["vs"][:N_tot*N]
ps = data["ps"][:N_tot*N]
us = us[:N_tot*N]


Rots_w, vs_w, ps_w = propagate_coriolis(us, Rots[0], vs[0], ps[0], dt, Delta_t, N, N_tot, 0*Omega_coriolis, 1, g, Rots)
Rots_wo, vs_wo, ps_wo = propagate_coriolis(us, Rots[0], vs[0], ps[0], dt, Delta_t, N, N_tot, Omega_coriolis, 1, g, Rots)
Rots_std, vs_std, ps_std = propagate_coriolis(us, Rots[0], vs[0], ps[0], dt, Delta_t, N, N_tot, Omega_coriolis, 2, g, Rots)


# plt.plot(SO3.to_rpy(Rots[N-1::N]))
# # plt.plot(SO3.to_rpy(Rots_wo))
# # plt.plot(SO3.to_rpy(Rots[::N])-SO3.to_rpy(Rots_wo))
# plt.show()

# err_Rot_wo = SO3.log(bmtm(Rots[N-1::N], Rots_wo).cuda()).cpu().norm(dim=1)
# err_Rot_std = SO3.log(bmtm(Rots[N-1::N], Rots_std).cuda()).cpu().norm(dim=1)
# err_Rot_w = SO3.log(bmtm(Rots[N-1::N], Rots_w).cuda()).cpu().norm(dim=1)
# plt.plot(err_Rot_wo)
# # plt.show()
# plt.plot(err_Rot_std)
# plt.plot(err_Rot_w)
# plt.figure()

# err_v_wo = (vs[N-1::N]-vs_wo).norm(dim=1)
# err_v_std = (vs[N-1::N]-vs_std).norm(dim=1)
# err_v_w = (vs[N-1::N]-vs_w).norm(dim=1)
# plt.plot(err_v_wo)
# plt.plot(err_v_std)
# plt.plot(err_v_w)
# plt.figure()

# err_p_wo = (ps[N-1::N]-ps_wo).norm(dim=1)
# err_p_std = (ps[N-1::N]-ps_std).norm(dim=1)
# err_p_w = (ps[N-1::N]-ps_w).norm(dim=1)
# plt.plot(err_p_wo)
# plt.plot(err_p_std)
# plt.plot(err_p_w)
# plt.show()

err_v_wo = (vs_w-vs_wo).norm(dim=1)
err_v_std = (vs_w-vs_std).norm(dim=1)

plt.plot(err_v_wo)
plt.plot(err_v_std)
# plt.figure()

# err_p_wo = (ps_w-ps_wo).norm(dim=1)
# err_p_std = (ps_w-ps_std).norm(dim=1)

# plt.plot(err_p_wo)
# plt.plot(err_p_std)
plt.show()