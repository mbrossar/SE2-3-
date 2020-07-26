import torch
from utils import *
from lie_group_utils import SO3, SE3_2

def f_Gamma(g, dt):
    """Compute Gamma preintegration"""
    Gamma = torch.eye(5)
    Gamma[:3, 3] = g*dt
    Gamma[:3, 4] = 1/2*g*(dt**2)
    return Gamma

def f_flux(T, dt):
    """Compute Phi (the flux) preintegration"""
    Phi = T.clone().view(-1, 5, 5)
    Phi[:, :3, 4] += Phi[:, :3, 3]*dt
    return Phi.squeeze()
