import os

import torch
from torch import nn
from tqdm.auto import tqdm
import torchaudio
import librosa


class MelSpectrogram(nn.Module):
    def __init__(self, config):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel


def get_data_to_buffer(train_config, mel_config):
    buffer = list()
    paths_to_wavs = os.listdir(train_config.wavs_path)

    wav_to_mel = MelSpectrogram(mel_config)

    for i in tqdm(range(len(paths_to_wavs))):
        wav_path = os.path.join(train_config.wavs_path, paths_to_wavs[i])
        wav, sr = torchaudio.load(wav_path)
        wav = wav.squeeze().double()

        # Expected shape is [B, T]
        mel = wav_to_mel(wav.unsqueeze(0)).squeeze(0)

        buffer.append(
            {
                'mel_spec': mel,
                'wav':wav
            }
        )

    return buffer