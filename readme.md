# DFlat
# 01/29/2024->01/30/2024. This repository is temporarily down while a new version is being uploaded.

## An End-to-End Design Framework for Diffractive Optics and Metasurface-Based Vision Systems
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Pytest Workflow](https://github.com/DeanHazineh/DFlat-pytorch/actions/workflows/pytest.yml/badge.svg?branch=dflat_v2.0.0)](https://github.com/DeanHazineh/DFlat-pytorch/actions/workflows/pytest.yml)
![Version](https://img.shields.io/badge/version-2.0.0-blue)

`D-Flat` is an auto-differentiable design framework for flat optics, specially geared to the design of dielectric metasurfaces for imaging and sensing tasks. This package was first introduced in a 2022 manuscript available at <a href="https://arxiv.org/abs/2207.14780" target="_blank">arxiv</a>. It was later published alongside our paper, <a href="https://deanhazineh.github.io/publications/Multi_Image_Synthesis/MIS_Home.html" target="_blank"> Polarization Multi-Image Synthesis with Birefringent Metasurfaces</a>, in the proceedings of the 2023 IEEE International Conference of Computational Photography (ICCP). 

D-Flat provides users with:
- A validated, auto-differentiable framework for field propagation, point-spread function calculations, and image rendering built on Pytorch.
- A growing set of pre-trained, efficient neural networks to model the optical response of metasurface cells (alongside the released datasets).
- A new and simple design architecture for adding your own datasets and training your own models.
- (Coming soon) An auto-differentiable field solver (RCWA) packaged in an easy to use module for building new datasets or optimizing small metasurfaces.
- (Coming later) A new set of modules for freeform cell design.

By treating optical layers in the same fashion as standard, differentiable neural layers, deep learning pipelines can be built to simultaneously optimize optical hardware and ML computational back-ends for the next generation of computational imaging devices.

## Version Notes (Version 2.0.0)
- This repository is the home for the new and maintained version of DFlat. It replaces DFlat-Tensorflow.
- Note that this package is no longer a direct port of pytorch-tensorflow but is a complete rewrite (re-)released in February 2024. 
- The structure of the software is completely revamped and the algorithms used--in particular for field propagation--is not the same as before. The original pytorch version (now deprecated) is archived as a seperate branch only for archival purposes.

## Usage and Documentation:
- Documentation and a new project page coming alongside tutorials in February 2024.
For developers and researchers,

  





