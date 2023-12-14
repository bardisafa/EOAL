## [EOAL: Entropic Open-set Active Learning](https://github.com/bardisafa/EOAL/) (AAAI 2024)
### Bardia Safaei, Vibashan VS, Celso M. de Melo, Vishal M. Patel
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

Active Learning (AL) aims to enhance the performance of deep models by selecting the most informative samples for annotation from a pool of unlabeled data. Despite impressive performance in closed-set settings, most AL methods fail in real-world scenarios where the unlabeled data contains unknown categories. Recently, a few studies have attempted to tackle the AL problem for the open-set setting. However, these methods focus more on selecting known samples and do not efficiently utilize unknown samples obtained during AL rounds. In this work, we propose an Entropic Open-set AL (EOAL) framework which leverages both known and unknown distributions effectively to select informative samples during AL rounds. Specifically, our approach employs two different entropy scores. One measures the uncertainty of a sample with respect to the known-class distributions. The other measures the uncertainty of the sample with respect to the unknown-class distributions. By utilizing these two entropy scores we effectively separate the known and unknown samples from the unlabeled data resulting in better sampling. Through extensive experiments, we show that the proposed method outperforms existing state-of-the-art methods on CIFAR-10, CIFAR-100, and TinyImageNet datasets.

![framework](figures/framework.png)

Table of Contents
=================

   * [Setup and Dependencies](#setup-and-dependencies)
   * [Usage](#usage)
      * [Train Active Domain Adaptation model](#train-active-domain-adaptation-model)
      * [Download data](#data-download)
      * [Pretrained checkpoints](#pretrained-checkpoints)
      * [Evaluation and plotting Results](#evaluation-and-plotting-results)
      * [Demo](#demo)
   * [Reference](#reference)
   * [License](#license)

## Setup and Dependencies

1. Create and activate a conda environment with Python 3.7 as follows: 
```
conda create -n EOAL python=3.7.16
conda activate EOAL
```
2. Install dependencies: 
```
pip install -r environment.txt
``` 
4. Modify the dataloader.py file in the torch.util.data.Dataloader source code as described [here](https://github.com/ningkp/LfOSA/issues/4).
   
## Run 

### CIFAR-10
Run ```python train.py``` to train an active adaptation model from scratch, by passing it appropriate arguments.

We include hyperparameter configurations to reproduce paper numbers on DIGITS and DomainNet as configurations inside the ```config``` folder. For instance, to reproduce DIGITS (SVHN->MNIST) results with CLUE+MME, run:

```
python train.py --load_from_cfg True \ 
                --cfg_file config/digits/clue_mme.yml \
                --use_cuda False
```

To run a custom train job, you can create a custom config file and pass it to the train script. Pass `--use_cuda False` if you'd like to train on CPU instead.

### CIFAR-100

Data for SVHN->MNIST is downloaded automatically via PyTorch. For DomainNet, follow the following steps:
1. Download the original dataset for the domains of interest from [this link](http://ai.bu.edu/M3SDA/) – eg. Clipart and Sketch.
2. Run: 
```
python preprocess_domainnet.py --input_dir <input_directory> \
                               --domains 'clipart,sketch' \
                               --output_dir 'data/'
```

## Reference

If you find this codebase useful in your research, please consider citing our paper:
```

```

## Acknowledgements

This code is built upon [LfOSA](https://github.com/ningkp/LfOSA) repository.
