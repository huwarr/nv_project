#!/bin/env bash
pip install -r requirements.txt
pip install gdown==4.5.4 --no-cache-dir

# Download HiFiGAN checkpoint
gdown https://drive.google.com/uc?id=1BhaVPY2wwpGWAT_rIPmGxjiBbR2eKmcu
