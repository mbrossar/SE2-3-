import torch
import os
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

def pload(*f_names):
    """Pickle load"""
    f_name = os.path.join(*f_names)
    with open(f_name, "rb") as f:
        pickle_dict = pickle.load(f)
    return pickle_dict

def pdump(pickle_dict, *f_names):
    """Pickle dump"""
    f_name = os.path.join(*f_names)
    with open(f_name, "wb") as f:
        pickle.dump(pickle_dict, f)

def mkdir(*paths):
    '''Create a directory if not existing.'''
    path = os.path.join(*paths)
    if not os.path.exists(path):
        os.mkdir(path)

def yload(*f_names):
    """YAML load"""
    f_name = os.path.join(*f_names)
    with open(f_name, 'r') as f:
        yaml_dict = yaml.load(f)
    return yaml_dict

def ydump(yaml_dict, *f_names):
    """YAML dump"""
    f_name = os.path.join(*f_names)
    with open(f_name, 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)

def bmv(mat, vec):
    """batch matrix vector product"""
    return torch.einsum('bij, bj -> bi', mat, vec)

def bbmv(mat, vec):
    """double batch matrix vector product"""
    return torch.einsum('baij, baj -> bai', mat, vec)

def bmtv(mat, vec):
    """batch matrix transpose vector product"""
    return torch.einsum('bji, bj -> bi', mat, vec)

def bmtm(mat1, mat2):
    """batch matrix transpose matrix product"""
    return torch.einsum("bji, bjk -> bik", mat1, mat2)

def bmmt(mat1, mat2):
    """batch matrix matrix transpose product"""
    return torch.einsum("bij, bkj -> bik", mat1, mat2)

def outer(vec1, vec2):
    """outer product"""
    return torch.einsum('i, j -> ij', vec1, vec2)

def bouter(vec1, vec2):
    """batch outer product"""
    return torch.einsum('bi, bj -> bij', vec1, vec2)

def btrace(mat):
    """batch matrix trace"""
    return torch.einsum('bii -> b', mat)

def axat(A, X):
    r"""Returns the product A X A^T."""
    return torch.einsum("ij, jk, lk->il", A, X, A)

def baxat(A, X):
    r"""Returns the product A X A^T. (batch)"""
    return torch.einsum("bij, bjk, blk->bil", A, X, A)

def isclose(mat1, mat2, tol=1e-10):
    """
    Check element-wise if two tensors are close within some tolerance.
    """
    return (mat1 - mat2).abs().lt(tol)

def bdot(vec1, vec2):
    """batch scalar product"""
    return torch.einsum('bs,bs->b', vec1, vec2)

def pltt(x):
    """Plot PyTorch tensor on CUDA and with grad"""
    plt.plot(x.detach().cpu())

def plts(x):
    """Plot and show PyTorch tensor on CUDA and with grad"""
    plt.plot(x.detach().cpu())
    plt.show()
    
    
def op1(A):
    """
    <<.>> operator.
    
    <<A>> = - tr(A) Id + A
    """
    return -torch.trace(A)*torch.eye(3) + A

def op2(A, B):
    """
    <<., .>> operator.
    
    <<A, B>> = <<A>> <<B>> + <<BA>>
    """
    return op1(A).mm(op1(B)) + op1(B.mm(A))

def cov(m):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.

    Returns:
        The covariance matrix of the variables.
    '''
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()

def fro_norm(cov1, cov2):
    tmp = cov1 - cov2
    return (tmp.t().mm(tmp)).trace().sqrt()

def savefig(path, axs, fig, name):
        if isinstance(axs, np.ndarray):
            for i in range(len(axs)):
                axs[i].grid()
                axs[i].legend()
        else:
            axs.grid()
            axs.legend()
        fig.tight_layout()
        fig.savefig(path + name + '.png')


def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result

def four_order(Sigma, Q):
        Sigma_pp = Sigma[:3, :3]
        Sigma_vp = Sigma[3:6, :3]
        Sigma_vv = Sigma[3:6, 3:6]
        Sigma_vr = Sigma[3:6, 6:9]
        Sigma_rp = Sigma[6:9, :3]
        Sigma_rr = Sigma[6:9, 6:9]

        Qpp = Q[:3, :3]
        Qvp = Q[3:6, :3]
        Qvv = Q[3:6, 3:6]
        Qvr = Q[3:6, 6:9]
        Qrp = Q[6:9, :3]
        Qrr = Q[6:9, 6:9]

        A1 = Sigma.new_zeros(9, 9)
        A1[:3, :3] = op1(Sigma_pp)
        A1[3:6, :3] = op1(Sigma_vp + Sigma_vp.t())
        A1[3:6, 3:6] = op1(Sigma_pp)
        A1[6:9, :3] = op1(Sigma_rp + Sigma_rp.t())
        A1[6:9, 6:9] = op1(Sigma_pp)

        A2 = Sigma.new_zeros(9, 9)
        A2[:3, :3] = op1(Qpp)
        A2[3:6, :3] = op1(Qvp + Qvp.t())
        A2[3:6, 3:6] = op1(Qpp)
        A2[6:9, :3] = op1(Qrp + Qrp.t())
        A2[6:9, 6:9] = op1(Qpp)

        Bpp = op2(Sigma_pp, Qpp)
        Bvv = op2(Sigma_pp, Qvv) + op2(Sigma_vp.t(), Qvp) +\
            op2(Sigma_vp, Qvp.t()) + op2(Sigma_vv, Qpp)
        Brr = op2(Sigma_pp, Qrr) + op2(Sigma_rp.t(), Qrp) +\
            op2(Sigma_rp, Qrp.t()) + op2(Sigma_rr, Qpp)
        Bvp = op2(Sigma_pp, Qvp.t()) + op2(Sigma_vp.t(), Qpp)
        Brp = op2(Sigma_pp, Qrp.t()) + op2(Sigma_rp.t(), Qpp)
        Bvr = op2(Sigma_pp, Qvr) + op2(Sigma_vp.t(), Qrp) +\
            op2(Sigma_rp, Qvp.t()) + op2(Sigma_vr, Qpp)

        B = Sigma.new_zeros(9, 9)
        B[:3, :3] = Bpp
        B[:3, 3:6] = Bvp.t()
        B[:3, 6:9] = Brp.t()
        B[3:6, :3] = B[:3, 3:6].t()
        B[3:6, 3:6] = Bvv
        B[3:6, 6:9] = Bvr
        B[6:9, :3] = B[:3, 6:9].t()
        B[6:9, 3:6] = B[3:6, 6:9].t()
        B[6:9, 6:9] = Brr

        return (A1.mm(Q) + Q.t().mm(A1.t()) + A2.mm(Sigma) + Sigma.t().mm(A2.t()))/12 + B/4