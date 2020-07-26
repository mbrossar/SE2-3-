

Associating Uncertainty to Extended Poses for on Lie Group IMU  Preintegration with Rotating Earth
==================================================================================================

This repo contains provides Python scripts that implement the major equations from the [paper]() mentioned above. The GTSAM fork related to the papier is available at this [url](https://github.com/mbrossar/gtsam). The repo also contains [supplementary material]() that provides detailed proofs along with comprehensive technical derivations of the paper.

Paper Overview [ArXiv paper]()
---------------------------------------------
The recently introduced matrix group $SE_2(3)$ provides a 5x5 matrix representation for the  orientation, velocity and position of an object  in the 3-D space, a triplet we call ``extended pose''. In the paper we build on this group to develop a theory to associate uncertainty with extended poses represented by 5x5 matrices. Our approach is particularly suited to describe how uncertainty propagates when the extended pose represents the state of an Inertial Measurement Unit (IMU). In particular it allows  revisiting the theory of IMU preintegration on manifold and  reaching a further theoretic level in this field. Exact preintegration formulas that account for rotating Earth, that is, centrifugal force and Coriolis force, are derived as a byproduct, and the factors are shown to be more accurate. The approach is validated through extensive simulations and applied to sensor-fusion where a loosely-coupled fixed-lag smoother fuses IMU and LiDAR on   one hour long   experiments using our experimental car. It shows how handling rotating Earth may be beneficial for long-term navigation within incremental smoothing algorithms.

Installation
-------------------
These scripts are based on the [PyTorch](https://pytorch.org/) library with CUDA installed for highly fast batch computation (It assumes the desktop is equipped with a GPU). It necessitates to install the following packages

```
pip install torch torchvision pyyaml matplotlib numpy scipy typing termcolor
```

The repo has been tested on a Ubuntu 16.04 desktop with 1.5 PyTorch version.

Description of the Scripts
------------------
* **intro.py** generates plots for comparing $SE_2(3)$ and $SO(3)$ distributions (Figure 1)
* **simple_propagation.py**	generates the plots for the example of propagation of an extended pose (Figure 2 and Figure 5)
* **fourth_order.py** generates the plot for comparing second- and fourth-order methods (Figure 3)
* **retraction.py**	generates the plots for comparing $SE_2(3)$ and $SO(3)$ distributions (Figure 4)
* **preintegration.py** generates the plots for the IMU preintegration comparison (Figure 7)
* **bias.py** generates plots for the bias update comparison (Figure 8)
* **coriolis.py** generates plots for the Coriolis comparison (Figure 9)
* **lie_group_utils.py** implements $SO(3)$ and $SE_2(3)$ related functions
* **preintegration_utils.py** contains functions for preintegration
* **numerical_test.py** compares numerical Jacobian and integration versus our analytical expressions related to $\Gamma$ factors and IMU increments.

The implementation is based with **batch** in the first dimension, e.g. 

```
xis = torch.randn(N, 3) # N is batch size
Rots = SO3.exp(xis) # Nx3x3 rotation matrices
```
contains $N$ rotation matrices. This allows really fast  :rocket: Monte-Carlo sampling.


GTSAM
---------------------
GTSAM is a C++ library that implements smoothing and mapping algorithms using factor-graphs. Our GTSAM fork of at these [url](https://github.com/mbrossar/gtsam) contains implementation for
* Bias update with Lie exponential coordinates
* Proposed rotating Earth and Coriolis effect preintegration
* Debug of the original rotating Earth and Coriolis effect preintegration

where we have modified the following files:

* **ManifoldPreintegration.cpp**
* **NavState.cpp**
* **PreintegrationBase.cpp**

## Citation

If you use this code in your research, please cite:

```
@article{brossard2020associating,
  author={M. {Brossard} and A. {Barrau} and P. {Chauchat} and S. {Bonnabel}},
  title = {Associating Uncertainty to Extended Poses for on Lie Group IMU  Preintegration with Rotating Earth},
  year = {2020}
}
```

## Authors

This code was written by the [Centre of Robotique](http://caor-mines-paristech.fr/en/home/) at the MINESParisTech, Paris, France.

[Martin
Brossard](mailto:martin.brossard@mines-paristech.fr), [Axel
Barrau](mailto:axel.barrau@safrangroup.com), [Paul Chauchat](mailto:paul.chauchat@isae-supaero.fr) and [Silvère
Bonnabel](mailto:silvere.bonnabel@mines-paristech.fr).