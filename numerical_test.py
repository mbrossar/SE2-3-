import torch
from utils import *
from lie_group_utils import SO3, SE3_2
import matplotlib.pyplot as plt
import numpy as np
import pickle
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=4)
from termcolor import cprint

def assert_almost_equal(a, b, name, TOL):
    val = (a-b).norm().item()
    if val > TOL:
        cprint("!!Test " + name + " FAILS (error: {:E}".format(val) + ")", 'red')
    else:
        cprint("Test " + name + " successes (error: {:E}".format(val) + ")", 'green')

def test_Gamma_factor():
    """
    Test that close-form expressions related to the Gamma factors (equations (82), (83), and (84) of the paper).
    """

    dt = 0.001 # really small sampling time
    N = 1000 # increment number
    t = N*dt
    Omega = torch.Tensor([0.2, 0.1, 0.3]).cuda() # random Earth rate
    Omega_skew = SO3.uwedge(Omega).cpu()
    Omega_incr = SO3.uexp(-Omega.cpu()*dt)
    g = torch.Tensor([2,4,9.81]) # gravity vector
    TOL = 1e-5

    # numerical integration
    # initialize factor
    Gamma_R = torch.eye(3)
    Gamma_v = torch.zeros(3)
    Gamma_p = torch.zeros(3)

    # integrate
    for n in range(N):
        Gamma_p = Gamma_p + (Gamma_v - Omega_skew.mv(Gamma_p))*dt
        Gamma_v = Gamma_v + (g - Omega_skew.mv(Gamma_v))*dt
        Gamma_R = Omega_incr.mm(Gamma_R)

    # close-form expression
    Gamma_R_cf = SO3.uexp(-Omega.cpu()*t).cpu()
    Gamma_v_cf = SO3.left_jacobian(-Omega.view(1,-1)*t).cpu().squeeze().mv(g * t)
    phi = Omega.norm().cpu()
    a =  ( t*phi*(t*phi).cos() - (t*phi).sin() ) / (phi**3)
    b = ((t*phi)**2-2*(t*phi).cos()-2*(t*phi)*(t*phi).sin()+2) / (2*(phi**4))
    A = (t**2)/2 * torch.eye(3) + a * Omega_skew + b * Omega_skew.mm(Omega_skew)
    Gamma_p_cf = A.mv(g)

    # compare numerical and analytical expressions
    assert_almost_equal(Gamma_R, Gamma_R_cf, 'Gamma_R', TOL)
    assert_almost_equal(Gamma_v, Gamma_v_cf, 'Gamma_v', TOL)
    assert_almost_equal(Gamma_p, Gamma_p_cf, 'Gamma_p', TOL)


def numerical_Jacobian(f, x, delta_x=1e-8):
    DfDx = torch.zeros(f(x).shape[0], x.shape[0])
    for i in range(x.shape[0]):
        xPlus = x.clone()
        xPlus[i] += delta_x
        xMoins = x.clone()
        xMoins[i] -= delta_x
        DfDx[:, i] = (f(xPlus)-f(xMoins))/(2*delta_x)
    return DfDx.cpu()



def test_Deltav_Jacobian_A(TOL=1e-5):
    """Check derivative of A, i.e. equation (74) of supplementary material w.r.t. gyro noise"""


    Deltat = 0.1 + torch.randn(1).abs().item()
    Omega = torch.randn(3).cuda()
    def f(Omega):
        phi = Omega.norm()
        return ((1-(phi*Deltat).cos())/(phi**2)).view(1)

    J_num = numerical_Jacobian(f, Omega)
    phi = Omega.norm()
    s = (phi*Deltat).sin()
    c = (phi*Deltat).cos()
    J_ana = (Omega.t() * (phi*Deltat*s - 2 + 2*c) / phi**4).cpu()
    assert_almost_equal(J_num, J_ana, 'Derivative A', TOL)



def test_Deltav_Jacobian_B(TOL=1e-5):
    """Check derivative of B, i.e. equation (75) of supplementary material w.r.t. gyro noise"""


    Deltat = 0.1 + torch.randn(1).abs().item()
    Omega = torch.randn(3).cuda()
    def f(Omega):
        phi = Omega.norm()
        return ((phi*Deltat-(phi*Deltat).sin())/(phi**3)).view(1)

    J_num = numerical_Jacobian(f, Omega)
    phi = Omega.norm()
    s = (phi*Deltat).sin()
    c = (phi*Deltat).cos()
    u = phi*Deltat
    J_ana = (Omega.t() * (-2*u - u*c + 3*s) / phi**5).cpu()
    assert_almost_equal(J_num, J_ana, 'Derivative B', TOL)



def test_Deltav_Jacobian(TOL=1e-5):
    """Check derivative of Delta v_i, i.e. equation (71) of supplementary material w.r.t. gyro noise"""


    Deltat = 0.3 + torch.randn(1).abs().item()
    Omega = torch.randn(3).cuda()
    a_i = torch.randn(3)
    def f(Omega):
        phi = Omega.norm()
        return SO3.left_jacobian(Omega.view(1,-1)*Deltat).cpu().squeeze().mv(a_i*Deltat)

    J_num = numerical_Jacobian(f, Omega)
    phi = Omega.norm()
    c = (phi*Deltat).cos()
    s = (phi*Deltat).sin()
    u = phi*Deltat

    a_i_skew = SO3.uwedge(a_i)
    Omega_skew = SO3.uwedge(Omega).cpu()

    A = (1-(phi*Deltat).cos())/(phi**2)
    B = (phi*Deltat-(phi*Deltat).sin())/(phi**3)
    A1 = -A * a_i_skew
    A2 = - B * ( Omega_skew.mm(a_i_skew) + SO3.uwedge(Omega_skew.mv(a_i)))
    DADphi = (Omega.t() * (phi*Deltat*s - 2 + 2*c) / phi**4).cpu()
    DBDphi = (Omega.t() * (-2*u - u*c + 3*s) / phi**5).cpu()
    A3 = outer(Omega_skew.mv(a_i), DADphi)
    A4 = outer(Omega_skew.mm(Omega_skew).mv(a_i), DBDphi)

    J_ana = A1 + A2 + A3 + A4
    assert_almost_equal(J_num, J_ana, 'Derivative Delta v_i', TOL)


def test_Deltap_Jacobian_a(TOL=1e-5):
    """Check derivative of a, i.e. equation (78) of supplementary material w.r.t. gyro noise"""


    Deltat = 0.1 + torch.randn(1).abs().item()
    Omega = torch.randn(3).cuda()
    def f(Omega):
        phi = Omega.norm()
        res = (Deltat*phi*(Deltat*phi).cos() - (Deltat*phi).sin() ) / (phi**3)
        return res.view(1)

    J_num = numerical_Jacobian(f, Omega)
    phi = Omega.norm()
    s = (phi*Deltat).sin()
    c = (phi*Deltat).cos()
    u = phi*Deltat
    u2 = u**2
    J_ana = (Omega.t() * (-u2 * s - 3*u*c + 3*s) / phi**5).cpu()
    assert_almost_equal(J_num, J_ana, 'Derivative a', TOL)



def test_Deltap_Jacobian_b(TOL=1e-5):
    """Check derivative of B, i.e. equation (79) of supplementary material w.r.t. gyro noise"""


    Deltat = 0.1 + torch.randn(1).abs().item()
    Omega = torch.randn(3).cuda()
    def f(Omega):
        phi = Omega.norm()
        res = ((Deltat*phi)**2-2*(Deltat*phi).cos()-2*(Deltat*phi)*(Deltat*phi).sin()+2) / (2*(phi**4))
        return res.view(1)

    J_num = numerical_Jacobian(f, Omega)
    phi = Omega.norm()
    s = (phi*Deltat).sin()
    c = (phi*Deltat).cos()
    u = phi*Deltat

    u2 = u**2
    J_ana = (Omega.t() * (u2 - u2*c -2*(u2-2*c-2*u*s+2)) / phi**6).cpu()
    assert_almost_equal(J_num, J_ana, 'Derivative B', TOL)


def test_Deltap_Jacobian(TOL=1e-5):
    """Check derivative of Delta p_i, i.e. equation (76) of supplementary material w.r.t. gyro noise"""


    Deltat = 0.3 + torch.randn(1).abs().item()
    Omega = torch.randn(3).cuda()
    a_i = torch.randn(3)
    def f(Omega):
        phi = Omega.norm()
        Omega_skew = SO3.uwedge(Omega).cpu()
        t = Deltat
        a =  (t*phi*(t*phi).cos() - (t*phi).sin() ) / (phi**3)
        b = ((t*phi)**2-2*(t*phi).cos()-2*(t*phi)*(t*phi).sin()+2) / (2*(phi**4))
        A = (t**2)/2 * torch.eye(3) + a * Omega_skew + b * Omega_skew.mm(Omega_skew)
        return A.mv(a_i)

    J_num = numerical_Jacobian(f, Omega)
    phi = Omega.norm()
    c = (phi*Deltat).cos()
    s = (phi*Deltat).sin()
    u = phi*Deltat
    t2 = u**2
    t = Deltat

    a_i_skew = SO3.uwedge(a_i)
    Omega_skew = SO3.uwedge(Omega).cpu()

    a = (t*phi*(t*phi).cos() - (t*phi).sin() ) / (phi**3)
    b = ((t*phi)**2-2*(t*phi).cos()-2*(t*phi)*(t*phi).sin()+2) / (2*(phi**4))
    A1 = -a * a_i_skew
    A2 = - b * ( Omega_skew.mm(a_i_skew) + SO3.uwedge(Omega_skew.mv(a_i)))
    DaDphi = (Omega.t() * (phi**-5) * (-(phi**2 * t**2) * s - 3*phi*t*c + 3*s)).cpu()
    DbDphi = (Omega.t() * (phi**-6) * (t2 - t2*c -2*((t*phi)**2-2*(t*phi).cos()-2*(t*phi)*(t*phi).sin()+2))).cpu()
    A3 = outer(Omega_skew.mv(a_i), DaDphi)
    A4 = outer(Omega_skew.mm(Omega_skew).mv(a_i), DbDphi)

    J_ana = A1 + A2 + A3 + A4
    assert_almost_equal(J_num, J_ana, 'Derivative Delta v_i', TOL)


def test_G(TOL=1e-3):
    """Check"""


    Deltat = 0.1 + 0.1*torch.randn(1).abs().item()
    omega = 0.1*torch.randn(3)
    acc = torch.randn(3)
    
    u = torch.cat((omega, acc, 0.5*acc*Deltat))*Deltat
    Ups = torch.eye(5)
    Ups[:3, :3] = SO3.uexp(u[:3])
    Ups[:3, 3] = u[3:6]
    Ups[:3, 4] = u[6:9]
    Jac = SO3.left_jacobian(u[:3].cuda().view(-1, 3)).cpu().squeeze()
    invJac = SO3.inv_left_jacobian(u[:3].cuda().view(-1, 3)).cpu().squeeze()
    
    def f(eta):
        omega_i = omega + eta[:3]
        acc_i = acc + eta[3:6]
        u = torch.cat((omega_i, acc_i, 0.5*acc_i*Deltat))*Deltat
        Ups_i = torch.eye(5)
        Ups_i[:3, :3] = SO3.uexp(u[:3])
        Ups_i[:3, 3] = u[3:6]
        Ups_i[:3, 4] = u[6:9]
        return SE3_2.ulog(Ups_i.inverse().mm(Ups).cuda()).cpu()

    J_num = numerical_Jacobian(f, torch.zeros(6))
    
    J_ana = torch.zeros(9, 6)
    J_ana[:3, :3] = -invJac*Deltat
    J_ana[3:6, 3:] = -Ups[:3, :3].t()*Deltat
    J_ana[6:9, 3:] = -0.5*Ups[:3, :3].t()*Deltat**2
    
    cov = torch.eye(6)
    sigma_omega = torch.randn(1).abs()
    sigma_acc = torch.randn(1).abs()
    cov[:3, :3] *= sigma_omega**2
    cov[3:6, 3:6] *= sigma_acc**2

    Q = J_ana.mm(cov).mm(J_ana.t())
        
    Q_ana = torch.zeros(9, 9)
    Q_ana[:3, :3] = (sigma_omega**2)*torch.eye(3)
    Q_ana[3:6, 3:6] = (sigma_acc**2)*torch.eye(3)
    Q_ana[6:9, 6:9] = 0.25*(sigma_acc**2)*torch.eye(3)*Deltat**2
    Q_ana[3:6, 6:9] = 0.5*(sigma_acc**2)*torch.eye(3)*Deltat
    Q_ana[6:9, 3:6] = Q_ana[3:6, 6:9].t()
    Q_ana = Q_ana * Deltat**2

    assert_almost_equal(J_num, J_ana, 'G_i constant global acceleration', TOL)
    assert_almost_equal(Q, Q_ana, 'Q constant global acceleration', TOL)

test_G()
test_Deltav_Jacobian_A()
test_Deltav_Jacobian_B()
test_Deltav_Jacobian()
test_Deltap_Jacobian_a()
test_Deltap_Jacobian_b()
test_Deltap_Jacobian()
test_Gamma_factor()