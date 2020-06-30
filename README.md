

Associating Uncertainty to Extended Poses for IMU Preintegration on Lie Group
===================================================================================

These Python scripts provide an implementation of the major equations from the paper mentioned above. 


Installation
-------------------
These scripts are based on the [PyTorch](https://pytorch.org/) with CUDA library for highly fast batch computation (It assumes the desktop is equipped with a GPU). It necessitate to install the following packages

```
pip install torch torchvision pyyaml matplotlib numpy scipy typing
```

Experiment Scripts
------------------		
* **intro.py**		generates plots for comparing SE2(3) and SO(3) distributions (Figure 1)
* **simple_propagation.py**		generates plots for the example of propagation of an extended pose (Figure 2)
* **second_vs_four_orders.py**		generates plots for comparing second- and fourth-order methods (Figure 3)
* **retraction.py**		generates plots for comparing SE2(3) and SO(3) distributions (Figure 4)
* **preintegration.py**		generates plots for the preintegration comparison (Figure 6)
* **bias.py**		generates plots for the bias update comparison (Figure 7)
* **coriolis.py**		generates plots for Coriolis comparison (Figure 8)



Math Implementations
--------------------

* **lie_group_utils.py**		SO(3) and SE2(3) related functions
* **preintegration_utils.py**		generates plots for second pose compound experiment in paper (figure 3)

The implementation is done with **batch** in the first dimension, e.g. 

```
xis = torch.randn(N, 3) # N is batch size
Rots = SO3.exp(xis) # Nx3x3 rotation matrices
```