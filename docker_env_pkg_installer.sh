#!/bin/bash

apt update
apt install ffmpeg libsm6 libxext6  -y

python -m pip install 'git+https://github.com/shivamsnaik/detectron2.git'
python -m pip install timm
python -m pip install opencv-python
