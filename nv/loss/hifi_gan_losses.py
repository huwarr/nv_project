import torch
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, out_real, out_fake):
        loss = 0.
        for i in range(len(out_real)):
            r = out_real[i]
            f = out_fake[i]

            loss_real = self.mse_loss(r, torh.ones_like(r))
            loss_fake = self.mse_loss(f, torh.zeros_like(f))

            loss += loss_real + loss_fake
        return loss


class DiscriminatorFeaturesLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, features_real, features_fake):
        loss = 0.
        for i in range(len(features_real)):
            rs = features_real[i]
            fs = features_fake[i]
            for j in range(len(rs)):
                r = rs[j]
                f = fs[j]
                loss += self.l1_loss(r, f)
        return loss * 2


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, out_fake):
        loss = 0.
        for out in out_fake:
            loss += self.mse_loss(out, torch.ones_like(out))
        return loss


class MelSpecLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.multiplier = 45

    def forward(self, melspec_real, melspec_fake):
        return self.l1_loss(melspec_real, melspec_fake) * self.multiplier