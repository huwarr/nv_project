import numpy as np
import torch
import torch.nn.functional as F

from nv.configs.all_configs import TrainConfig, MelSpectrogramConfig


def pad_1D_tensor(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])
    return padded


def pad_2D_tensor(inputs, maxlen=None, pad_value=0):
    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)), value=pad_value)
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])
    return output


def reprocess_tensor(batch, cut_list):
    mel_specs = [batch[ind]["mel_spec"] for ind in cut_list]
    wavs = [batch[ind]["wav"] for ind in cut_list]

    # we need pad_value
    mel_config = MelSpectrogramConfig()
    mel_specs = pad_2D_tensor(mel_specs, pad_value=mel_config.pad_value)
    wavs = pad_1D_tensor(wavs)

    out = {
        "mel_spec": mel_specs,
        "wav": wavs
    }
    return out


def collate_fn(batch):
    train_config = TrainConfig()

    len_arr = np.array([d["text"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = batchsize // train_config.batch_expand_size

    cut_list = list()
    for i in range(train_config.batch_expand_size):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(train_config.batch_expand_size):
        output.append(reprocess_tensor(batch, cut_list[i]))
    return output