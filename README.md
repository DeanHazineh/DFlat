# DFlat-pytorch
## This repository is the home for the new and maintained version of DFlat. It replaces DFlat-Tensorflow. 
Some improvements in this new software release include the following:
- The standard MLP is replaced with the SIREN model (sinusoidal activation functions) which significantly improve coordinate-mlp convergence and is an alternative to tancik's fourier features strategy
- The default datatype is changed from float64 to float32 substantially improving memory usage and efficiency. To ensure accuracy of calculations, the fourier layer also adopts mixed precision (switching to float64 for only the minimal parts of the calculation that require it). 
- For the time being, the built-in rcwa solver has been removed. Alongside a new paper to be released in the near future, DFlat will incorporate a new strategy for freeform metasurface optimization. This change reflects a new direction/vision for DFlat's usage
- We also add several improvements to the usability and code structuring. Writing your optimized metasurface designs to a gdspy file ready for fabrication is now much more straight-forward and easy.
- 
<div align="center">
  <img src=/docs/imgs/DFlat_Long.png alt="Dflat" width="500"/>
</div>
<div align="center">
  <img src=/docs/imgs/autoGDS_metalens.png alt="Dflat" width="500"/>
</div>

# An End-to-End Design Framework for Diffractive Optics and Metasurface-Based Vision Systems
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

`D-Flat` is an auto-differentiable design framework for flat optics, specially geared to the design of dielectric metasurfaces for imaging and sensing tasks. This package was officially released alongside our paper,  <a href="https://deanhazineh.github.io/publications/Multi_Image_Synthesis/combined_paper.pdf" target="_blank"> Polarization Multi-Image Synthesis with Birefringent Metasurfaces</a>, published in the proceedings of the 2023 IEEE International Conference of Computational Photography (ICCP). The package is further documented and discussed in the manuscript available on <a href="https://arxiv.org/abs/2207.14780" target="_blank">arxiv</a>. If you use this package, please cite the ICCP paper (See below for details). 

D-Flat provides users with:
- A validated, auto-differentiable framework for optical field propagation and rendering built on pytorch
- Pre-trained, efficient neural models to describe the optical response of metasurface cells
- An auto-differentiable field solver (RCWA) that is easy to use and call

By treating optical layers in the same fashion as standard, differentiable neural layers, deep learning pipelines can be built to simultaneously optimize optical hardware and ML computational back-ends for the next generation of computational imaging devices.

DFlat is also available on tensorflow at (<a href="https://github.com/DeanHazineh/DFlat-tensorflow/tree/main" target="_blank">DFlat-tensorflow</a>). Long-term support will only be provided for the pytorch version of the software.   

