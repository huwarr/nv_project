import os
import itertools
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler  import OneCycleLR
from tqdm.auto import tqdm

from nv.configs.all_configs import MelSpectrogramConfig, HiFiGANConfig, TrainConfig
from nv.utils.data_utils import get_data_to_buffer
from nv.dataset.buffer_dataset import BufferDataset
from nv.collate_fn.collate import collate_fn
from nv.model.hifi_gan import Generator, MPD, MSD
from nv.loss.hifi_gan_losses import DiscriminatorLoss, DiscriminatorFeaturesLoss, GeneratorLoss, MelSpecLoss
from nv.logger.wandb_logger import WanDBWriter
from nv.utils.data_utils import MelSpectrogram
from get_wav import run_full_synthesis


# fix seed for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# define configs
mel_config = MelSpectrogramConfig()
model_config = HiFiGANConfig()
train_config = TrainConfig()


# define dataloader
buffer = get_data_to_buffer(train_config, mel_config)
dataset = BufferDataset(buffer)

training_loader = DataLoader(
    dataset,
    batch_size=train_config.batch_expand_size * train_config.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=0
)

# training essentials
generator = Generator(model_config)
generator = generator.to(train_config.device)
generator.train()
msd = MSD(model_config)
msd = msd.to(train_config.device)
msd.train()
mpd = MPD(model_config)
mpd = mpd.to(train_config.device)
mpd.train()

discriminator_loss = DiscriminatorLoss()
features_loss = DiscriminatorFeaturesLoss()
generator_loss = GeneratorLoss()
melspec_loss = MelSpecLoss()

current_step = 0

optimizer_g = torch.optim.AdamW(
    generator.parameters(), 
    lr=train_config.learning_rate, 
    betas=(train_config.beta_1, train_config.beta_2)
)
optimizer_d = torch.optim.AdamW(
    itertools.chain(msd.parameters(), mpd.parameters()), 
    lr=train_config.learning_rate, 
    betas=(train_config.beta_1, train_config.beta_2)
)

# вот так можно было задать расписание из авторской реализации:
# scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
#     optimizer_g, gamma=train_config.lr_decay
# )
# но мы им больше не пользуемся, это просто пример на память :)

# logger
logger = WanDBWriter(train_config)

# training loop
tqdm_bar = tqdm(total=train_config.epochs * len(training_loader) * train_config.batch_expand_size - current_step)

wav_to_mel = MelSpectrogram(mel_config).to(train_config.device)

for epoch in range(train_config.epochs):
    for i, batchs in enumerate(training_loader):
        for j, db in enumerate(batchs):
            current_step += 1
            tqdm_bar.update(1)
            
            logger.set_step(current_step)

            wav_targets = db["wav"].float().unsqueeze(1).to(train_config.device)
            mel_specs = db["mel_spec"].float().to(train_config.device)

            # Run generator
            wav_fake = generator(mel_specs)
            mel_fake = wav_to_mel(wav_fake.squeeze(1))

            # Update discriminators
            optimizer_d.zero_grad()
            # MPD
            out_real, out_fake, _, _ = mpd(wav_targets, wav_fake.detach())
            loss_mpd = discriminator_loss(out_real, out_fake)
            # MSD
            out_real, out_fake, _, _ = msd(wav_targets, wav_fake.detach())
            loss_msd = discriminator_loss(out_real, out_fake)
            loss_d = loss_mpd + loss_msd
            loss_d.backward()
            optimizer_d.step()

            # Update generator
            optimizer_g.zero_grad()
            loss_mel = melspec_loss(mel_specs, mel_fake)

            _, out_fake_mpd, features_real_mpd, features_fake_mpd = mpd(wav_targets, wav_fake)
            _, out_fake_msd, features_real_msd, features_fake_msd = mpd(wav_targets, wav_fake)

            loss_features_mpd = features_loss(features_real_mpd, features_fake_mpd)
            loss_features_msd = features_loss(features_real_msd, features_fake_msd)
            loss_g = generator_loss(out_fake_mpd) + generator_loss(out_fake_msd)

            loss_g += loss_features_mpd + loss_features_msd + loss_mel
            loss_g.backward()
            optimizer_g.step()

            logger.add_scalar("generator loss", loss_g.detach().cpu().item())
            logger.add_scalar("discriminator loss", loss_d.detach().cpu().item())

            if current_step % train_config.save_step == 0:
                torch.save(
                    {'generator': generator.state_dict(), 'optimizer': optimizer_g.state_dict()}, 
                    os.path.join(train_config.checkpoint_path, 'generator_%d.pth.tar' % current_step))
                torch.save(
                    {'msd': msd.state_dict(), 'mpd': mpd.state_dict(), 'optimizer': optimizer_d.state_dict()}, 
                    os.path.join(train_config.checkpoint_path, 'discriminator_%d.pth.tar' % current_step))
                print("save model at step %d ..." % current_step)

                run_full_synthesis(checkpoint_path=os.path.join(train_config.checkpoint_path, 'generator_%d.pth.tar' % current_step), logger=logger)


# save checkpoint of the trained model
torch.save(
    {'generator': generator.state_dict(), 'optimizer': optimizer_g.state_dict()},
    'generator.pth.tar'
)

run_full_synthesis(checkpoint_path='generator.pth.tar', logger=logger)

logger.finish_wandb_run()