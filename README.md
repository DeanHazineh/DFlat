# DFlat-pytorch
## This repository is the home for the new and maintained version of DFlat. It replaces DFlat-Tensorflow. 
Some improvements in this new software release include the following:
- The standard MLP is replaced with the SIREN model (sinusoidal activation functions) which significantly improve coordinate-mlp convergence and is an alternative to tancik's fourier features strategy
- The default datatype is changed from float64 to float32 substantially improving memory usage and efficiency. To ensure accuracy of calculations, the fourier layer also adopts mixed precision (switching to float64 for only the minimal parts of the calculation that require it). 
- For the time being, the built-in rcwa solver has been removed. Alongside a new paper to be released in the near future, DFlat will incorporate a new strategy for freeform metasurface optimization. This change reflects a new direction/vision for DFlat's usage
- We also add several improvements to the usability and code structuring. Writing your optimized metasurface designs to a GDS file ready for fabrication is now much more straight-forward and easy.
- Installation is made easier. Datafiles will now be hosted in dropbox and will be automatically installed and unpacked to the right locations when the install function is called. This makes the package light-weight and easier to distribute--including a future release on the PyPI package index. 

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

`D-Flat` is an auto-differentiable design framework for flat optics, specially geared to the design of dielectric metasurfaces for imaging and sensing tasks. This package was officially released alongside our paper,  <a href="https://deanhazineh.github.io/publications/Multi_Image_Synthesis/combined_paper.pdf" target="_blank"> Polarization Multi-Image Synthesis with Birefringent Metasurfaces</a>, published in the proceedings of the 2023 IEEE International Conference of Computational Photography (ICCP). The package is further documented and discussed in an earlier pre-print manuscript available on <a href="https://arxiv.org/abs/2207.14780" target="_blank">arxiv</a>. If you use this package, please cite the ICCP paper (See below for details). 

D-Flat provides users with:
- A validated, auto-differentiable framework for optical field propagation and rendering built on pytorch
- Pre-trained, efficient neural models to describe the optical response of metasurface cells
- An easy platform for sharing metasurface libraries 

By treating optical layers in the same fashion as standard, differentiable neural layers, deep learning pipelines can be built to simultaneously optimize optical hardware and ML computational back-ends for the next generation of computational imaging devices.

An older version of DFlat is also available on tensorflow at (<a href="https://github.com/DeanHazineh/DFlat-tensorflow/tree/main" target="_blank">DFlat-tensorflow</a>). Long-term support will only be provided for this pytorch version of the software.   

## Installation
To install this software, run the following code at the terminal to clone and run the build: 
```
git clone https://github.com/DeanHazineh/DFlat-pytorch.git
pip install .
```
NOTE: As a courtesy, be aware that the install function in setup.py will execute code to download files, unzip, move, and delete temporary folders. If you do not want this functionality, comment out the lines that override the standard install behavior and run the setup_execute_get_data.py script yourself. 

## Example Code
You can test Dflat on google collab. Provided are some example scripts:
- <a href="https://colab.research.google.com/drive/18v2JYvcYnciRF1XgFGvdZDo8Ep5fRV6H?usp=sharing" target="_blank">Demo_polychromatic_focusing_nanofins.ipynb</a>


## Credits and Acknowledgements:
If you utilize DFlat or included data sets for your own work, please cite it by copying:

```
@INPROCEEDINGS{Hazineh2023,
  Author = {Dean Hazineh and Soon Wei Daniel Lim and Qi Guo and Federico Capasso and Todd Zickler},
  booktitle = {2023 IEEE International Conference on Computational Photography (ICCP)}, 
  Title = {Polarization Multi-Image Synthesis with Birefringent Metasurfaces},
  Year = {2023},
}
```

## Contact:
This repository is intended to be accessible and community driven. It may not be fully error-proof.
If you have improvements, fixes, or contributions, please branch and initiate a merge request to master (or email me)!
If you are interested in joining the team or working on helping to develop DFlat or applications, feel free to reach out. 

For any questions, functionality requests, or other concerns, don't hesitate to contact me at dhazineh@g.harvard.edu. 

