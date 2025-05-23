#!/bin/bash
set -e

# Env Setup.
apt install python3.7-venv
python3.7 -m venv ".venv"
source .venv/bin/activate

echo "##############################"
python --version
echo "##############################"

pip install torch==1.7.0 torchvision==0.8.1 numpy==1.19.2 scipy==1.5.2 pillow==8.0.1 tqdm==4.50.2 pytorch-lightning==1.2.5 efficientnet-pytorch==0.7.0 fvcore==0.1.2.post20201122
pip install matplotlib pyquaternion opencv-python

# Git config.
git config --global user.email "cedricxie@gmail.com"
git config --global user.name "Yuesong.xie"

wget 'https://github.com/wayveai/fiery/releases/download/v1.0/fiery.ckpt'
