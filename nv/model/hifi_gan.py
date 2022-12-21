import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm


class ResBlock(nn.Module):
    def __init__(self, n_channels, kernel_size, dilations):
        super().__init__()

        self.layers = []
        self.skip_connections = []
        self.activation = nn.LeakyReLU(0.1)

        for i in range(len(dilations)):
            dilation = dilations[i]
            self.layers.append(
                weight_norm(
                        nn.Conv1d(
                        n_channels, 
                        n_channels, 
                        kernel_size=kernel_size, 
                        dilation=dilation, 
                        padding=(kernel_size * dilation - dilation) // 2
                    )
                )
            )
            self.skip_connections.append(
                weight_norm(
                    nn.Conv1d(
                        n_channels, 
                        n_channels, 
                        kernel_size=kernel_size, 
                        dilation=1, 
                        padding=(kernel_size - 1) // 2
                    )
                )
            )

        self.layers = nn.ModuleList(self.layers)
        self.skip_connections = nn.ModuleList(self.skip_connections)
        self.__init_weights__()
    
    def __init_weights__(self):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.01)
        self.apply(init_func)
    
    def remove_weight_norm(self):
        for i in range(len(self.layers)):
            remove_weight_norm(self.layers[i])
            remove_weight_norm(self.skip_connections[i])

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](self.activation(x)) + self.skip_connections[i](self.activation(x))
        return x


class MRF(nn.Module):
    def __init__(self, n_channels, kernel_sizes, dilations):
        super().__init__()
        self.blocks = []

        for i in range(len(kernel_sizes)):
            self.blocks.append(
                ResBlock(n_channels, kernel_sizes[i], dilations[i])
            )
        
        self.blocks = nn.ModuleList(self.blocks)
    
    def remove_weight_norm(self):
        for block in self.blocks:
            block.remove_weight_norm()

    def forward(self, x):
        out = None
        for block in self.blocks:
            if out is None:
                out = block(x)
            else:
                out += block(x)
        return out # / len(self.blocks)    # in source code the authors take mean instead of sum, as they claim in the papaer


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_block = weight_norm(
            nn.Conv1d(
                config.in_channels, 
                config.upsample_hidden_dim, 
                kernel_size=7, 
                padding=3
            )
        )

        self.up_samples = []
        self.mrfs = []
        self.activation = nn.LeakyReLU(0.1)

        cur_channels = config.upsample_hidden_dim
        for i in range(len(config.upsample_kernel_sizes)):
            k_s = config.upsample_kernel_sizes[i]
            stride = config.upsample_kernel_sizes[i] // 2
            self.up_samples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        cur_channels, 
                        cur_channels // 2, 
                        kernel_size=k_s, 
                        stride=stride, 
                        padding=(k_s - stride) // 2
                    )
                )
            )
            self.mrfs.append(
                MRF(
                    cur_channels // 2,
                    config.res_blocks_kernel_sizes,
                    config.res_blocks_dilations
                )
            )
            cur_channels //= 2
        
        self.out_block = weight_norm(
            nn.Conv1d(
                cur_channels, 
                1, 
                kernel_size=7, 
                padding=3
            )
        )

        self.up_samples = nn.ModuleList(self.up_samples)
        self.mrfs = nn.ModuleList(self.mrfs)

        self.__init_weights__(self.up_samples)
        self.__init_weights__(self.in_block)
        self.__init_weights__(self.out_block)
    
    def __init_weights__(self, block):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.01)
        block.apply(init_func)

    def remove_weight_norm(self):
        remove_weight_norm(self.in_block)
        remove_weight_norm(self.out_block)
        for i in range(len(self.up_samples)):
            remove_weight_norm(self.up_samples[i])
            self.mrfs[i].remove_weight_norm()

    def forward(self, x):
        x = self.in_block(x)
        for i in range(len(self.up_samples)):
            x = self.up_samples[i](self.activation(x))
            x = self.mrfs[i](x)
        x = self.out_block(self.activation(x))
        return F.tanh(x)


class PeriodDiscriminator(nn.Module):
    def __init__(self, period, channel_sizes, kernel_size, stride):
        super().__init__()

        self.period = period
        self.activation = nn.LeakyReLU(0.1)
        
        self.in_block = weight_norm(
            nn.Conv2d(
                1,
                channel_sizes[0],
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=((kernel_size - 1) // 2, 0)
            )
        )

        self.layers = []
        for i in range(len(channel_sizes) - 1):
            self.layers.append(
                weight_norm(
                    nn.Conv2d(
                        channel_sizes[i],
                        channel_sizes[i + 1],
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1),
                        padding=((kernel_size - 1) // 2, 0)
                    )
                )
            )
        
        self.out_block = weight_norm(
            nn.Conv2d(
                channel_sizes[-1],
                1,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0)
            )
        )

        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        # reshape
        t = x.shape[2]
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(x.shape[0], x.shape[1], t // self.period, self.period)

        # 2D Convs
        x = self.activation(self.in_block(x))
        features = [x]
        for layer in self.layers:
            x = self.activation(layer(x))
            features.append(x)
        x = self.out_block(x)
        features.append(x)
        return torch.flatten(x, 1, -1), features


class MPD(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.layers = []
        for period in config.mpd_periods:
            self.layers.append(
                PeriodDiscriminator(
                    period,
                    config.mpd_channel_sizes,
                    config.mpd_kernel_size,
                    config.mpd_stride
                )
            )
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, y):
        out_real, out_fake = [], []
        features_real, features_fake = [], []
        for layer in self.layers:
            out, features = layer(x)
            out_real.append(out)
            features_real.append(features)
            out, features = layer(y)
            out_fake.append(out)
            features_fake.append(features)

        return out_real, out_fake, features_real, features_fake


class ScaleDiscriminator(nn.Module):
    def __init__(
        self, channel_sizes, kernel_sizes, strides, groups, paddings, is_first=False
    ):
        super().__init__()

        self.activation = nn.LeakyReLU(0.1)
        norm = spectral_norm if is_first else weight_norm

        self.in_block = norm(
            nn.Conv1d(
                1,
                channel_sizes[0],
                kernel_size=15,
                padding=7
            )
        )

        self.layers = []
        for i in range(len(channel_sizes) - 1):
            self.layers.append(
                norm(
                    nn.Conv1d(
                        channel_sizes[i],
                        channel_sizes[i + 1],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        padding=paddings[i],
                        groups=groups[i]
                    )
                )
            )
        
        self.out_block = norm(
            nn.Conv1d(
                channel_sizes[-1],
                1,
                kernel_size=3,
                padding=1
            )
        )

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        x = self.activation(self.in_block(x))
        features = [x]
        for layer in self.layers:
            x = self.activation(layer(x))
            features.append(x)
        x = self.out_block(x)
        features.append(x)
        return torch.flatten(x, 1, -1), features


class MSD(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = []
        n_layers = 3
        for i in range(n_layers):
            self.layers.append(
                ScaleDiscriminator(
                    config.msd_channel_sizes, 
                    config.msd_kernel_sizes, 
                    config.msd_strides, 
                    config.msd_groups,
                    config.msd_paddings,
                    is_first=(i == 0)
                )
            )
        self.layers = nn.ModuleList(self.layers)

        self.poolings = nn.ModuleList(
            [
                # first sub discriminator operates on raw audio
                nn.Identity(),
                nn.AvgPool1d(4, 2, padding=2),
                nn.AvgPool1d(4, 2, padding=2)
            ]
        )

    def forward(self, x, y):
        out_real, out_fake = [], []
        features_real, features_fake = [], []
        for i in range(len(self.layers)):
            layer = self.layers[i]
            pool = self.poolings[i]

            out, features = layer(pool(x))
            out_real.append(out)
            features_real.append(features)
            out, features = layer(pool(y))
            out_fake.append(out)
            features_fake.append(features)

        return out_real, out_fake, features_real, features_fake