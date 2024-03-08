#!/bin/bash

# fetch collected data repo
git clone https://github.com/hand-orthosis/collected_data.git

# delete the placeholder collected data folder and put in the real one
rm -rf data/collected_data/ && mv collected_data data/

conda create -n meta-emg python=3.10 pip
conda activate meta-emg
conda install pandas scipy matplotlib numpy 
pip install tqdm hydra-core higher wandb

# linux cuda 12.1 but command may need to be slightly different depending on your machine
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia