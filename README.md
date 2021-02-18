# CRaWl (Convolutional Neural Networks for Random Walks)

This repository contains the code for training and testing CRaWl networks.

#### Installation:
```
unzip CRaWl.zip
cd CRaWl
conda create --name crawl_env python=3.8
conda activate crawl_env
conda install pytorch=1.7.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
```
Note that we use cuda 10.2 by default. 


#### Experiments

The `scripts` directory provides bash scripts to execute both training and testing for all of our main experiments.
For example, run the following commands to train and test 5 CRaWl models on ZINC:

```
bash scripts/train_zinc.sh
bash scripts/test_zinc.sh
```
The datasets will be automatically downloaded when running the training scripts.

We set random seeds for model initialization, batch generation and the random walks.
The scripts use the same random seeds that were used to obtain the reported results in the paper.
Note that our implementation uses scatter operations that are non-deterministic for numerical reasons.
Training twice with the same seeds will not yield identical models, since the models will diverge through the training steps.
We aim to solve this in future versions.