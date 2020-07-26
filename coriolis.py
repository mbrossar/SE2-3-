import torch
from utils import *
from lie_group_utils import SO3, SE3_2
import matplotlib.pyplot as plt
import numpy as np
import pickle
torch.set_default_dtype(torch.float64)
from preintegration_utils import f_flux
from scipy.signal import savgol_filter

def f_Gamma(g, dt, omega_coriolis):
    """Compute Gamma preintegration with Coriolis forces"""
    vec = torch.cat((omega_coriolis, g, g))*dt
    tmp = SE3_2.uexp(vec.cuda()).cpu()[:4, :4] # for taking left-Jacobian
    Omega = SO3.uwedge(omega_coriolis)
    Omega2 = Omega.mm(Omega)
    ang = omega_coriolis.norm()
    if ang == 0: # without Coriolis
        mat = (dt**2)/2*torch.eye(3)
    else:
        a = (dt*ang * (dt*ang).cos() -(dt*ang).sin()) / (ang**3)
        b = (-dt*ang*(dt*ang).sin() - (dt*ang).cos() + 1 + ((dt*ang)**2)/2) /(ang**4)
        mat = (dt**2)/2*torch.eye(3) + a*Omega + b*Omega2
    Gamma = torch.eye(5)
    Gamma[:3, :3] = tmp[:3, :3]
    Gamma[:3, 3] = tmp[:3, 3]
    Gamma[:3, 4] = tmp[:3, :3].mm(mat).mv(g)
    return Gamma


def integrate(Upsilon, Rot, v, p, dt, N, Omega_coriolis, g):
    """One step Integration"""
    Gamma = f_Gamma(g, dt*N, Omega_coriolis)
    T0 = torch.eye(5)
    T0[:3, :3] = Rot
    T0[:3, 3] = v
    T0[:3, 4] = p
    Phi = f_flux(T0, dt*N)
    T = Gamma.mm(Phi.mm(Upsilon))
    Rot = T[:3, :3]
    v = T[:3, 3]
    p = T[:3, 4]
    return Rot, v, p

def integrate_imu(us, dt, N):
    """Compute IMU preintegration measurement"""
    Upsilon = torch.eye(5)
    Upsilon_i = torch.eye(5)
    Omega = SO3.exp(us[:N, :3].cuda()*dt).cpu()
    for i in range(N):
        Upsilon_i[:3, :3] = Omega[i]
        Upsilon_i[:3, 3] = us[i, 3:6]*dt
        Upsilon_i[:3, 4] = 1/2*us[i, 3:6]*(dt**2)
        Upsilon = f_flux(Upsilon, dt).mm(Upsilon_i)
    return Upsilon

def propagate_coriolis(us, Rot0, v0, p0, dt, Delta_t, N, N_tot, Omega_coriolis, g, Rots_gt):
    """Integrate with Coriolis"""
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
        Rots[i], vs[i], ps[i] = integrate(Upsilon, Rots[i-1], vs[i-1], ps[i-1], dt, N, Omega_coriolis, g)

    Rots = SO3.normalize(Rots.cuda()).cpu()
    return Rots, vs, ps

def compute_input(data, dt, g, Omega_coriolis, path):
    """Compute IMU measurement from ground-truth"""
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
    accs = bmtv(Rots[:-1], -g + tmp1[:-1] + tmp2[:-1] + 1/dt*(vs[1:]-vs[:-1]))

    us[1:, :3] = omegas
    us[1:, 3:6] = accs
    data["vs"] = vs
    data["us"] = us
    return data


# load ground-truth position and input
path = "figures/coriolis.p"
data = pload(path)
dt = 0.02 # (s)
Delta_t = 5
latitude = np.pi/180*48.7
earth_rate = 7.292115e-5 # rad/s
Omega_coriolis = earth_rate*torch.Tensor([0*np.cos(latitude), 0, -np.sin(latitude)]) # NED
g = torch.Tensor([0, 0, -9.81])
data = compute_input(data, dt, g, Omega_coriolis, path)

N = int(Delta_t/dt)
us = data["us"]
N_tot = int(us.shape[0]/N)
Rots = data["Rots"][:N_tot*N]
Rots[::N] = SO3.normalize(Rots[::N].cuda()).cpu()
vs = data["vs"][:N_tot*N]
ps = data["ps"][:N_tot*N]
us = us[:N_tot*N]


Rots_wo, vs_wo, ps_wo = propagate_coriolis(us, Rots[0], vs[0], ps[0], dt, Delta_t, N, N_tot, torch.zeros_like(Omega_coriolis), g, Rots)
Rots_w, vs_w, ps_w = propagate_coriolis(us, Rots[0], vs[0], ps[0], dt, Delta_t, N, N_tot, Omega_coriolis, g, Rots)

err_v_wo = (vs_w-vs_wo).norm(dim=1)
err_v_w = (vs_w-vs_w).norm(dim=1) # Method is benchmark
# smooth error due to ground-truth differentiation
err_v_wo = savgol_filter(err_v_wo.numpy(), 51, 3)
t = Delta_t/60 * np.linspace(0, err_v_wo.shape[0], err_v_wo.shape[0])
plt.plot(t, err_v_wo, 'red')
plt.plot(t, err_v_w, 'green')
plt.legend(["w/o Coriolis", "proposed"])
plt.xlabel('$t$ (min)')
plt.xlim(0, t[-1])
plt.ylim(0)
plt.grid()
plt.ylabel('Velocity error (m/s)')
plt.show()

# res = np.zeros((t.shape[0], 3))
# res[:, 0] = t
# res[:, 1] = err_v_wo
# res[:, 2] = err_v_w.numpy()
# res = res[::2]
# np.savetxt('figures/coriolis.txt', res, comments="", header = "t wo w")


