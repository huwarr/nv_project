import os

import torch
import torchaudio

from nv.configs.all_configs import MelSpectrogramConfig, HiFiGANConfig, TrainConfig
from nv.model.hifi_gan import Generator
from nv.utils.data_utils import MelSpectrogram


def get_data(train_config, mel_config):
    paths = sorted(os.listdir(train_config.samples_path))
    wav_to_mel = MelSpectrogram(mel_config)

    data_list = []

    for path in paths:
        wav_path = os.path.join(train_config.samples_path, path)
        wav, sr = torchaudio.load(wav_path)
        wav = wav.squeeze().double()
        mel = wav_to_mel(wav.unsqueeze(0)).squeeze(0)

        data_list.append(
            {
                'mek': mel,
                'path': path
            }
        )

    return data_list


def run_full_synthesis(checkpoint_path='generator.pth.tar', logger=None):
    train_config = TrainConfig()
    model_config = HiFiGANConfig()
    mel_config = MelSpectrogramConfig()

    model = Generator(model_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda:0')['model'])
    model = model.eval()
    model = model.to(train_config.device)

    data_list = get_data(train_config, mel_config)
    os.makedirs("results", exist_ok=True)

    for i in range(len(data_list)):
        mel = data_list[i]['mel']
        path = os.path.join('results', data_list[i]['path'])

        wav = model(mel.unsqueeze(0))
        torchaudio.save(path, wav, mel_config.sr)

        if logger is not None:
            logger.add_audio(data_list[i]['path'][:-4], wav.float(), sample_rate=mel_config.sr)


if __name__ == '__main__':
    run_full_synthesis()