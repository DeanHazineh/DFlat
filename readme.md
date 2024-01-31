<div align="center">
  <img src=/docs/DFlat_Long.png alt="Dflat" width="500"/>
</div>
<div align="center">
  <img src=/docs/autoGDS_metalens.png alt="Dflat" width="500"/>
</div>

# An End-to-End Design Framework for Diffractive Optics and Metasurface-Based Vision Systems
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Pytest Workflow](https://github.com/DeanHazineh/DFlat-pytorch/actions/workflows/pytest.yml/badge.svg?branch=dflat_v2.0.0)](https://github.com/DeanHazineh/DFlat-pytorch/actions/workflows/pytest.yml)
![Version](https://img.shields.io/badge/version-2.0.0-blue)

`D-Flat` is an auto-differentiable design framework for flat optics, specially geared to the design of dielectric metasurfaces for imaging and sensing tasks. This package was first introduced in a 2022 manuscript available at <a href="https://arxiv.org/abs/2207.14780" target="_blank">arxiv</a>. It was later published alongside our paper, <a href="https://deanhazineh.github.io/publications/Multi_Image_Synthesis/MIS_Home.html" target="_blank"> Polarization Multi-Image Synthesis with Birefringent Metasurfaces</a>, in the proceedings of the 2023 IEEE International Conference of Computational Photography (ICCP). It was rewritten in 2024 with Version 2.0.

D-Flat provides users with:
- A validated, auto-differentiable framework for field propagation, point-spread function calculations, and image rendering built on Pytorch.
- A growing set of pre-trained, efficient neural networks to model the optical response of metasurface cells (alongside the released datasets).
- A new and simple design architecture for adding your own datasets and training your own models.
  
By treating optical layers in the same fashion as standard, differentiable neural layers, deep learning pipelines can be built to simultaneously optimize optical hardware and ML computational back-ends for the next generation of computational imaging devices.

## Version Notes (Version 2.0.0)
- This repository is the home for the new and maintained version of DFlat. It replaces DFlat-Tensorflow.
- Note that this package is no longer a direct port of pytorch-tensorflow but is a complete rewrite (re-)released in February 2024. 
- The structure of the software is completely revamped and the algorithms used--in particular for field propagation--are not the same as before. The original pytorch version (now deprecated) is archived and kept as a branch.
- (Coming soon) Accessibility via downloading from the PyPI index.
- (Coming soon) An auto-differentiable field solver (RCWA) packaged in an easy to use module for building new datasets or optimizing small metasurfaces.
- (Coming later) A new set of modules for freeform cell design.

## Installation 
### (a) Install and run DFlat locally:
Install the local repository to your venv by entering the following in terminal:
```
git clone https://github.com/DeanHazineh/DFlat
python setup.py install
```
Note that the setup.py file will automatically download pre-trained model checkpoints. No part of this repostiory requires the raw data unless you want to re-train or continue training some models. To fetch the metasurface library data, you may run the bash script in terminal:
```
./download_raw_data.sh
```
If bash in unavailable, then you may download the zipped data files <a href="https://www.dropbox.com/scl/fi/efzz37tlejkkplo7pe7vs/data.zip?rlkey=malv67btexvfhkyhbiasgrai0&dl=1" target="_blank">here</a>. You would then need to manually unzip and place the files in the metasurface/data/ folder. 

### (b) Use Dflat on google collab:
Note that DFlat can be easily installed and used in the cloud on Google Collab if desired by executing the above in the jupyter notebook. This is beneficial if you do not have local access to a gpu. 
```
!git clone https://github.com/DeanHazineh/DFlat
%cd /content/DFlat
!python setup.py install
!./download_raw_data.sh  # (optionally if you need access to raw data used to train the models)
```

## Usage and Documentation
Detailed documentation for the rewritten package and a new project page will be released at the end of February 2024. For now, we highlight two resources for developers and researchers:
 - The script used to train neural models can be found at `scripts/trainer.ipynb`.
 - A simple demo of the workflow can be found at `scripts/demo.ipynb`. 

Google collab versions of current examples can be accesssed online at the links:
- link
- link

List of packaged Neural Models and corresponding data:

  
## Credits and Acknowledgements:
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
 * Lucidrain - Siren-pytorch
 * Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega - Pyhank
 * Shane Colburn - RCWA_TF


