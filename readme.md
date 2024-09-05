<div align="center">
  <img src=/docs/DFlat_Long.png alt="Dflat" width="500"/>
</div>
<div align="center">
  <img src=/docs/autoGDS_metalens.png alt="Dflat" width="500"/>
</div>

# An End-to-End Design Framework for Diffractive Optics and Metasurface-Based Vision Systems
## Auto-differentiable RCWA, Field Propagation, Rendering, Neural Surrogate Models

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

![Version](https://img.shields.io/badge/version-4.0.0-blue)
[![PyPI version](https://badge.fury.io/py/dflat-opt.svg)](https://badge.fury.io/py/dflat-opt)
[![Pytest Workflow](https://github.com/DeanHazineh/DFlat-pytorch/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/DeanHazineh/DFlat-pytorch/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/DeanHazineh/DFlat/graph/badge.svg?token=3J0LPUJ1OX)](https://codecov.io/gh/DeanHazineh/DFlat)

`DFlat` is an auto-differentiable design framework for flat optics, specially geared to the design of dielectric metasurfaces for imaging and sensing tasks. This package was first introduced in a 2022 manuscript available at <a href="https://arxiv.org/abs/2207.14780" target="_blank">arxiv</a>. It was later published alongside our paper, <a href="https://deanhazineh.github.io/publications/Multi_Image_Synthesis/MIS_Home.html" target="_blank"> Polarization Multi-Image Synthesis with Birefringent Metasurfaces</a>, in the proceedings of the 2023 IEEE International Conference of Computational Photography (ICCP). It was rewritten in 2024 with Version 2.0 and subsequent releases.

DFlat provides users with:

- A validated, auto-differentiable framework for field propagation, point-spread function calculations, and image rendering built on Pytorch.
- A growing set of pre-trained, efficient neural networks to model the optical response of metasurface cells (alongside the released datasets).
- A new and simple design architecture for adding your own datasets and training your own models.
- A autodifferentiable field solver (RCWA) to optimize structures on cells or generate new libraries

By treating optical layers in the same fashion as standard, differentiable neural layers, deep learning pipelines can be built to simultaneously optimize optical hardware and ML computational back-ends for the next generation of computational imaging devices.


## Installation 

You can install this package from the PyPI index simply via: 
```
pip install dflat_opt
```
You  may also find it beneficial to work directly with the repository in editable mode via:
```
git clone https://github.com/DeanHazineh/DFlat
pip install -e .
```

## Version Notes
Updated on July 12, 2024 | V3 -> V4
- Documentation is still in development :o
- (v4) An autodifferentiable RCWA field solver class is added. 
- (v3) Memory problems? Gradient checkpointing is added throughout reducing memory when using DFlat.
- (v3) You can now donwload this package from the PyPi Index!
- (v3) Datasets and pre-trained models are now downloaded when called insetad of during install. Models are now initialized by their name instead of by paths to config files.
- (v2) This repository is the home for the new and maintained version of DFlat. It replaces DFlat-Tensorflow. Note that this package is no longer a direct port of DFlat-tensorflow but is a complete rewrite (re-)released in February 2024. I recommend switching! The structure of the software is completely revamped and the algorithms used throughout are not the same as before. The initial pytorch port (now deprecated) is archived and kept as a branch.

## Usage and Documentation

Detailed documentation for the rewritten package and a new project page will be released in the future. For now, we highlight two resources for developers and researchers:
- A script used to train neural models can be found in `scripts/trainer.ipynb`. The model architecture is specified in the config.yaml file and you can play around with editing it. 
- A simple demo of the workflow can be found at `scripts/demo.ipynb`.

### Pretrained Models and datasets

List of pretrained neural models inlcuded at this time are noted below. You can see the dataset used, training parameters, and model info in the respective config.yaml files when the model is downloaded after calling `load_optical_model(MODEL_NAME)`.
If you would like to contribute your own data or models to this open-source repository, please email me and I can add it with acknowledgements. We aim to make this repository a centralized location for sharing meta-atom datasets and pre-trained neural representations. 
| Si3N4 Models | TiO2 Models |
| :---: | :---: |  
| | Nanocylinders_TiO2_U200H600 |  
| Nanocylinders_Si3N4_U250H600 | Nanocylinders_TiO2_U250H600 |  
| Nanocylinders_Si3N4_U300H600 | Nanocylinders_TiO2_U300H600 |  
| Nanocylinders_Si3N4_U350H600 | Nanocylinders_TiO2_U350H600 |  
| | Nanoellipse_TiO2_U350H600 |  
| | Nanofins_TiO2_U350H600 |

The datasets will be downloaded and can be played with simply by calling the classes in dflat/metasurface/datsets. For example, when you call `Nanoellipse_TiO2_U350nm_H600nm()`, the corresponding dataset will be downloaded and you can play around with the data and visualizations within that class.

## Contact:

This repository is intended to be accessible and community driven. It may not be fully error-proof.
If you have improvements, fixes, or contributions, please branch and initiate a merge request to master (or email me)!
For any questions, functionality requests, or other concerns, don't hesitate to contact me at dhazineh@g.harvard.edu.

## Credits and Acknowledgements:
This work was funded in part by the NSF and SONY.
If you utilize DFlat or included data sets for your own work, please consider citing it by using either:

```
@INPROCEEDINGS{10233735,
  author={Hazineh, Dean and Lim, Soon Wei Daniel and Guo, Qi and Capasso, Federico and Zickler, Todd},
  booktitle={2023 IEEE International Conference on Computational Photography (ICCP)},
  title={Polarization Multi-Image Synthesis with Birefringent Metasurfaces},
  year={2023},
  pages={1-12},
  doi={10.1109/ICCP56744.2023.10233735}}
```

```
@misc{hazineh2022dflat,
      title={D-Flat: A Differentiable Flat-Optics Framework for End-to-End Metasurface Visual Sensor Design},
      author={Dean S. Hazineh and Soon Wei Daniel Lim and Zhujun Shi and Federico Capasso and Todd Zickler and Qi Guo},
      year={2022},
      eprint={2207.14780},
      archivePrefix={arXiv},
      primaryClass={physics.optics}
}
```

This work has pulled inspiration from and/or benefitted from previous open-source contributions including:
- Lucidrain - Siren-pytorch
- Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega - Pyhank
- Shane Colburn - RCWA_TF

