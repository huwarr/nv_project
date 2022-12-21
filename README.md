# Neural Vocoder project

An attempt to implement and train `HiFiGAN`.

[checkpoint](...)

**synthesised audio:** in `generated samples` folder

## Installation guide

First, clone this repository to get access to the code:

`git clone https://github.com/huwarr/nv_project.git`

`cd nv_project`

## Synthesising audio

Run `setup.sh` script to download requirements and all the necessary files for synthesisng audio, including checkpoint:

`sh setup.sh`

Run python file to synthesis audio samples using checkpoint:

`python get_wav.py`

When everything is done, you can view samples in `results` folder :)

## Training

Firts, load training data with running the script:

`sh load_data.sh`

Install dependencies:

`pip install -r requirements.txt`

Run training script:

`python train.py`

You will see intermideate checkpoints of the model in `model_new` folder, the final checkpoint as `generator.pth.tar` in the root of this repository, and samples, synthesised with the final checkpoint, in `results` folder.

## Sources

1. [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/pdf/2010.05646.pdf)
