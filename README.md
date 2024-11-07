[![DOI](https://zenodo.org/badge/698535974.svg)](https://doi.org/10.5281/zenodo.14050016)

# AtomAI: Reusability

[**Original AtomAI repository (up to date)**](https://github.com/pycroscopy/atomai)

[**Original AtomAI repository (fork used)**](https://github.com/pycroscopy/atomai/tree/657ee4f0ea0efaeeb541cf0c5b76377a64a72f06)

## Files contributed

The following files are the main files contributed by this work, the path of the files are provided below. (__init__ and other small changes are not included in this list)

- atomai/utils/instance_seg_utils.py: "Introduced utilities for Instance Segmentation"
- atomai/transforms/imaug.py: "Fixed image augmentation to access all flips and rotations"
- atomai/trainers/trainer.py: "Introduced Early Stopping and LSTM trainer for ImSpec"
- atomai/nets/ed.py: "Signal LSTM introduced"		
- atomai/models/segmentor.py: "Introduced Binary Automatic-Thresholding and changes for Early Stopping "
- atomai/models/loaders.py: "Changes made for Binary Automatic-Thresholding"
- atomai/models/imspec.py: "ImSpecTrainerLSTM introduced" 

- Mat2Spec/
	- Mat2Spec/Mat2Spec.py
	- Mat2Spec/SinkhornDistance.py
	- Mat2Spec/data.py
	- Mat2Spec/file_setter.py
	- Mat2Spec/pytorch_stats_loss.py
	- Mat2Spec/utils.py

- examples/
	- examples/notebooks/EarlyStopping_HRTEM_pv.ipynb
	- examples/notebooks/EarlyStopping_LBFO_pv.ipynb
	- examples/notebooks/Instance_Segmentation.ipynb
	- examples/notebooks/LSTM_EarlyStopping_ImSpec.ipynb
	- examples/notebooks/Mat2Spec_ImSpec.ipynb
	
	




## To reproduce our results, please refer to the following Jupyter notebooks:

### Example 1: LBFO dataset demo
[Implementation of Updated Deep Convolutional Neural Network for Atom Finding using Early stopping](https://github.com/Pragalbhv/AtomAI_Reusability/blob/master/examples/notebooks/EarlyStopping_LBFO_pv.ipynb)

## Example 2: HRTEM dataset (Au/CdSe/Combined) demo
[Implementation of Updated DCNN equipped with Early stopping and automatic binary-thresholding for Nano-particle Segmentation](https://github.com/Pragalbhv/AtomAI_Reusability/blob/master/examples/notebooks/EarlyStopping_HRTEM_pv.ipynb)

## Example 3: Instance Segmentation: Additive Manufacturing dataset demo
[Implementation of Semantic Segmentation from Semantic Segmentation maps for optical micrographs of Additive manufacturing dataset](https://github.com/Pragalbhv/AtomAI_Reusability/blob/master/examples/notebooks/Instance_Segmentation.ipynb)

## Example 4: ImSpec demo
[Implementation of Updated Demo for ImSpec on the STEM EELS dataset using Early Stopping and LSTM Decoder](https://github.com/Pragalbhv/AtomAI_Reusability/blob/master/examples/notebooks/LSTM_EarlyStopping_ImSpec.ipynb)

## Example 5: Mat2Spec demo
[Implementation of the modified Mat2Spec for ImSpec on the STEM EELS dataset](https://github.com/Pragalbhv/AtomAI_Reusability/blob/master/examples/notebooks/Mat2Spec_ImSpec.ipynb)

### The following configurations were used for training
The parameters like training cycles, patience, and the number of augmented images created are modified so as to run in a Google Colab notebook without any premium resources. 
The training cycles were set to 5000 and patience to 500, and the number of images produced was ~ 2500.

One only needs to run the Colab notebooks as is. However, if one wants to test it on a personal machine, they can download the necessary datafiles for Example 1 and 2 [here.](https://drive.google.com/drive/folders/1S6pfS0t-rJ9U4EvNl0J3bB5wolJ9yRIt?usp=sharing)

The additive manufacturing dataset can be obtained [here.](https://drive.google.com/file/d/1s2_9Mmha7q6CcM5LRTyE1EbpbEE4Er5k/view?usp=drive_link)




### Credits and rights for the dataset used in each file belong to the respective entities:
  - LBFO: C. Nelson, A. Ghosh, M. Ziatdinov and S. Kalinin V, (Zenodo, 2021).
  - Au &CdSe: Groschner, C., Choi, C., & Scott, M. (2021). Machine Learning Pipeline for Segmentation and Defect Identification from High-Resolution Transmission Electron Microscopy Data. Microscopy and Microanalysis, 27(3), 549-556. doi:10.1017/S1431927621000386
  - Additive Manufacturing Dataset: Can be provided upon request to the authors.
  - STEM-EELS: Roccapriore, Kevin M. and Ziatdinov, Maxim and Cho, Shin Hum and Hachtel, Jordan A. and Kalinin, Sergei V. (2021). Predictability of Localized Plasmonic Responses in Nanoparticle Assemblies. doi:10.1002/smll.202100181

