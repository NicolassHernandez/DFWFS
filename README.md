# DFWFS
Official implementation for "[Generalized Deep Fourier-ased Wavefront Sensing](https://preprints.opticaopen.org/articles/preprint/Deep_Optics_Preconditioner_for_Modulation-free_Pyramid_Wavefront_Sensing/23812041)"

![](figures/Scheme_a.pdf)

![](figures/Scheme_b.pdf)


# Installation
- install anaconda (https://www.anaconda.com/products/distribution)
- on anaconda prompt (windows) or terminal (linux) create enviroment with the .yml file:
```
conda env create -f env.yml -n my_environment
conda activate my_environment
```

## HowTo

- You must first generate data using the generators provided in the pytorch folder.

  Example for modulation 0, resolution 128 and get 35 Zernike decomposition of each phasemap:

  ```
  python DataGeneratorCuda.py --modulation 0 --samp 2 --D 8 --nPxPup 128 --zModes [2,36]

  ```
  by default the script create 100 training phases and zernike decompositions and 10 for validation. 
  
- Then for training you have to use the same parameters generated:

  ```
  python E2E_main.py --modulation 0 --samp 2 --D 8 --nPxPup 128 --zModes [2,36] --batchSize 1

  ```
  
  check the E2E_main script parser help for extra parameters like noise and pyramid shape. If more GPU's are available, you can use ``` --gpu 0,1,N ``` to load the process with data paralelization, or run mutiple instances on each GPU.

When a training instance is run, the results are saved in ```./train_results/"expname"/```. You can set up the experiment name with ```--experimentName```. In this folder, the following parameters are saved:


# Reproducing Results
All the figures generated in the research paper were obtained using the scripts in the MATLAB folder. Please ensure that you set up the path to the Diffractive Element (saved as a .mat file) in the first lines of each script you wish to run. Here is a list of available figures that can be reproduced:

```
Paper_Fig4.m
Paper_Fig5_6.m
Paper_Fig7_8.m
Paper_Fig9.m
Paper_Fig10.m
Paper_Fig12.m
```
However it is highly recommended to use python scripts for figures 5, 7, 9 and 10 on ```Detach_OOMAO/Pytorch/figure_calculators/``` to compute the results and then using the MATLAB scripts to plot the figures.

# Citation
If you find our project useful, please cite:

```
@article{guzman2024deep,
  title={Deep optics preconditioner for modulation-free pyramid wavefront sensing},
  author={Guzm{\'a}n, Felipe and Tapia, Jorge and Weinberger, Camilo and Hern{\'a}ndez, Nicol{\'a}s and Bacca, Jorge and Neichel, Benoit and Vera, Esteban},
  journal={Photonics Research},
  volume={12},
  number={2},
  pages={301--312},
  year={2024},
  publisher={Optica Publishing Group}
}
