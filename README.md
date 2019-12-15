# e4040-2019Fall-Project


This repo serves as the final project of the course **ECBM4040 at Columbia University** for students - 

- Ujwal Dinesha ud2130
- Jeswanth Yadagani jy3012
- Ruturaj Rajendra Nene rn2494

### Recommended System Requirements to run this repo:
- CPU: Intel Skylake (16 vCPU)
- RAM: 104 GB
- GPU: Nvidia Tesla P100

### Instructions for running
The two main notebooks that need to be run to train the spectrum of models (7 models in total, with PeeleNet at one end of the spectrum and DenseNet-41 on the other) on CIFAR10 and ImageNet respectively are:
1. Run_this_cifar10
2. Run_this_imagenet

*Note: Whenever we mention ImageNet in this repo, we are referring to the subset of ImageNet (containing 198 classes with almost 500 images per class) that was downloaded using https://github.com/mf1024/ImageNet-Datasets-Downloader*

### File structure
```
Repo:.
|   README.md   
|   requirements.txt (env dependancies)
|   Run_this_cifar10.ipynb (train models on CIFAR10)
|   Run_this_imagenet.ipynb (train models on ImageNet)
|   
+---imagenet_download (tool for downloading ImageNet dataset courtesy of Martin Frolovs)
|       classes_in_imagenet.csv
|       downloader.py
|       imagenet_class_info.json
|       prepare_stats.py
|       README.md
|       requirements.txt
|       words.txt
|       
+---models (saved models)
|   +---cifar
|   |       m0.h5
|   |       m1.h5
|   |       m3.h5
|   |       m4.h5
|   |       m5.h5
|   |       m6.h5
|   |       
|   \---imagenet
|           m0.h5
|           m1.h5
|           m3.h5
|           m4.h5
|           m5.h5
|           m6.h5
|           
\---utils (utility functions)
        imagenet_utils.py (download and process ImageNet dataset)
        layer_utils.py (building blocks for model architecture)
        model_utils.py (formation of models)
```        

