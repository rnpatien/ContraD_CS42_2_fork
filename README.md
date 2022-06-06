# Extensions to ContraD package supporting dammageGAN and imbalance dataset 

This repository contains the code for reproducing the paper:
** Do we need a new generative model? A practical comparison between VAE and GANs variants Sydney university Capstone CS48-2 
by Yanbing Liu, yliu6286@uni.sydney.edu.au, Yuqing Chen, yche4082@uni.sydney.edu.au, Tianyu Wang, twan8010@uni.sydney.edu.au, Xintong Chu,xchu5428@uni.sydney.edu.au
Moyang Chen,  mche5278@uni.sydney.edu.au, Ziheng Pan,  zpan0520@uni.sydney.edu.au, Robert Patience,  rpat3029@uni.sydney.edu.au


## Overview

![Teaser](./resources/concept1.jpg)

*An overview of Contrastive Discriminator (ContraD).
In contraD the representation is not learned from the discriminator loss (L_dis), 
but from two contrastive losses (L+_con and L-_con), each is for the real and fake samples, respectively.
The damage GAN users the same approach but users pruning of the network to highlight minor features*

## Dependencies

Currently, the following environment has been confirmed to run the code:
* `python >= 3.6`
* `pytorch >= 1.6.0` (See [https://pytorch.org/](https://pytorch.org/) for the detailed installation)
* `tensorflow-gpu == 1.14.0` to run `test_tf_inception.py` for FID/IS evaluations
* Other requirements can be found in `environment.yml` (for conda users) or `environment_pip.txt` (for pip users)
```
#### Install dependencies via conda.
# The file also includes `pytorch`, `tensorflow-gpu=1.14`, and `cudatoolkit=10.1`.
# You may have to set the correct version of `cudatoolkit` compatible to your system.
# This command creates a new conda environment named `contrad`.
conda env create -f environment.yml

#### Install dependencies via pip.
# It assumes `pytorch` and `tensorflow-gpu` are already installed in the current environment.
pip install -r environment_pip.txt
```

### Preparing datasets

By default, the code assumes that all the datasets are placed under `data/`. 
You can change this path by setting the `$DATA_DIR` environment variable.

**CIFAR-10/100** can be automatically downloaded by running any of the provided training scripts.   

The structure of `$DATA_DIR` should be roughly like as follows:   
```
$DATA_DIR
├── cifar-10-batches-py   # CIFAR-10

```

## Scripts

### Training Scripts

We provide training scripts to reproduce the results in `train_*.py`, as listed in what follows:

| File | Description |
| ------ | ------ |
| [train_gan.py](train_gan.py) |  Train a GAN model other than StyleGAN2. DistributedDataParallel supported. |


The samples below demonstrate how to run each script to train GANs with ContraD.
One can modify `CUDA_VISIBLE_DEVICES` to further specify GPU number(s) to work on.

```
# SNDCGAN + ContraD on CIFAR-10
CUDA_VISIBLE_DEVICES=0 python train_gan.py configs/gan/cifar10/c10_b512.gin sndcgan \
--mode=contrad --aug=simclr --use_warmup

```
### Experiments CIFAR-10
```
# G: SNDCGAN / D: SNDCGAN 
python train_gan.py configs/gan/cifar10/c10_b64.gin sndcgan --mode=std

```



### Testing Scripts

* The script [test_gan_sample.py](test_gan_sample.py) generates and saves random samples from 
  a pre-trained generator model into `*.jpg` files. For example,
  ```
  CUDA_VISIBLE_DEVICES=0 python test_gan_sample.py PATH/TO/G.pt sndcgan --n_samples=10000
  ```
  will load the generator stored at `PATH/TO/G.pt`, generate `n_samples=10000` samples from it,
  and save them under `PATH/TO/samples_*/`.

* The script [test_gan_sample_cddls.py](test_gan_sample_cddls.py) additionally takes the discriminator, and 
  a linear evaluation head obtained from `test_lineval.py` to perform class-conditional cDDLS. For example,
  ```
  CUDA_VISIBLE_DEVICES=0 python test_gan_sample_cddls.py LOGDIR PATH/TO/LINEAR.pth.tar sndcgan
  ```
  will load G and D stored in `LOGDIR`, the linear head stored at `PATH/TO/LINEAR.pth.tar`,
  and save the generated samples from cDDLS under `LOGDIR/samples_cDDLS_*/`.

* The script [test_lineval.py](test_lineval.py) performs linear evaluation for a given 
  pre-trained discriminator model stored at `model_path`:
  ```
  CUDA_VISIBLE_DEVICES=0 python test_lineval.py PATH/TO/D.pt sndcgan
  ```

* The script [test_tf_inception.py](test_tf_inception.py) computes Fréchet Inception distance (FID) and
  Inception score (IS) with TensorFlow backend using the original code of FID available at https://github.com/bioinf-jku/TTUR.
  `tensorflow-gpu <= 1.14.0` is required to run this script. It takes a directory of generated samples 
  (e.g., via `test_gan_sample.py`) and an `.npz` of pre-computed statistics:
  ```
  python test_tf_inception.py PATH/TO/GENERATED/IMAGES/ PATH/TO/STATS.npz --n_imgs=10000 --gpu=0 --verbose
  ```
  A pre-computed statistics file per dataset can be either found in http://bioinf.jku.at/research/ttur/, 
  or manually computed - you can refer [`third_party/tf/examples`](third_party/tf/examples) for the sample scripts to this end.
  


