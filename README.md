# DFlat-pytorch
## Coming In A Few Weeks. A full port of DFlat-tensorflow to Pytorch with substantial speed and memory improvements!

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

