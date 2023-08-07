# CaLLTiF: Causal Discovery from Whole Brain fMRI Data

## Introduction

This repository contains Python codes used for data processing and simulations in the article titled "Whole-Brain Causal Discovery Using fMRI" by Arab, F., Ghassami, A., Jamalabadi, H., Peters, M. A. K., & Nozari, E. (2023). The repository includes two sets of experiments: one using simulated fMRI data and another using real resting-state fMRI data from the Human Connectome Project (HCP) dataset. Additionally, this package introduces the new proposed causal discovery method called CaLLTiF, designed based on the PCMCI causal discovery method and is suitable for whole-brain causal discovery.

## Reference

 The referenced paper for this work is "Whole-Brain Causal Discovery Using fMRI" by Arab, F., Ghassami, A., Jamalabadi, H., Peters, M. A. K., & Nozari, E. (2023). You can find the paper on bioRxiv.

## External Packages and Files

Before running the codes in this repository, ensure that the following Python packages are installed:

- numpy
- scipy
- matplotlib
- pandas
- networkx
- nibabel
- seaborn
- importlib
- nilearn
- sklearn
- netplotbrain
- mpl_toolkits.axes_grid1

Additionally, the following external packages are required:

- Tigramite package, available at https://github.com/jakobrunge/tigramite
- Causal discovery package, available at https://github.com/ckassaad/causal_discovery_for_time_series
- DGLearn method, available at https://github.com/syanga/dglearn
- DYNOTEARS method, available at https://github.com/quantumblacklabs/causalnex/blob/develop/causalnex/structure/dynotears.py
- NTS-NOTEARS method available at https://github.com/xiangyu-sun-789/NTS-NOTEARS
- Causal Command, available at https://www.ccd.pitt.edu/tools/

Additionally, the following files need to be downloaded and made available to the Python codes (either placed in the same directory or specify the path in the code):

- The parcel specifications for the Schaefer 100x7 parcellation, available at https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_100Parcels_7Networks_order.txt
- Color codes for the 7 resting-state networks, available at https://github.com/jimmyshen007/NeMo/blob/master/resource/Atlasing/Yeo_JNeurophysiol11_MNI152/Yeo2011_7Networks_ColorLUT.txt
- Parcel coordinates for 100 parcels in the Schaefer 100x7 parcellation atlas, available at https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv
- Parcel specifications for the Tian subcortical atlas, available at https://github.com/yetianmed/subcortex/blob/master/Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S1_3T_label.txt
- Parcel coordinates for Tian subcortical parcels, available at https://github.com/yetianmed/subcortex/blob/master/Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S1_3T_COG.txt

## Data Availability

The Python codes in the "HCP_Causality" directory assume that preprocessed HCP S1200 rsfMRI time series are available in an 'HCP' subdirectory. The data is not included in this distribution but is publicly available at https://db.humanconnectome.org. The preprocessing pipeline is described in the Methods section of the referenced paper.

Similarly, the Python codes in the "Simulated_fMRI_Causality" directory assume that simulated fMRI time series are available. These are not included in this distribution but are publicly available at https://github.com/cabal-cmu/Feedback-Discovery.

## Contact Information

For any questions or issues related to this repository, please contact Fahimeh Arab at farab002@ucr.edu.

