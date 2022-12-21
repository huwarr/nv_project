import torch
from dataclasses import dataclass

@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251

@dataclass
class HiFiGANConfig:
    # Generator
    in_channels = 80    # n_mels

    upsample_kernel_sizes = [16, 16, 4, 4]
    upsample_hidden_dim = 512

    res_blocks_kernel_sizes = [3, 7, 11]
    res_blocks_dilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

    # MPD
    mpd_kernel_size = 5
    mpd_stride = 3
    mpd_periods = [2, 3, 5, 7, 11]
    mpd_channel_sizes = [32, 128, 512, 1024, 1024]

    # MSD
    msd_kernel_sizes = [41, 41, 41, 41, 41, 5]
    msd_strides = [2, 2, 4, 4, 1, 1]
    msd_groups = [4, 16, 16, 16, 16, 1]
    msd_paddings = [20, 20, 20, 20, 20, 2]
    msd_channel_sizes = [128, 128, 256, 512, 1024, 1024, 1024]


@dataclass
class TrainConfig:
    checkpoint_path = "./model_new"
    logger_path = "./logger"
    wavs_path = './data/LJSpeech-1.1/wavs'
    samples_path = './samples'
    
    wandb_project = 'hifi_gan'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 16
    epochs = 30

    learning_rate = 2e-4
    beta_1 = 0.8
    beta_2 = 0.99
    lr_decay = 0.999
    
    save_step = 500

    batch_expand_size = 1