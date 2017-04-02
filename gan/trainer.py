import os
import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from models import Encoder, Decoder, Discriminator, weight_init
from torch.autograd import Variable


class gan_trainer:
    def __init__(self, z_dim=32, h_dim=128, filter_num=64, channel_num=3,
                 lr=1e-3):
        # Encoder, decoder, discriminator
        self.encoder = Encoder(
            z_dim, h_dim=h_dim, filter_num=filter_num, channel_num=channel_num)
        self.encoder.apply(weight_init)

        self.decoder = Decoder(
            z_dim, filter_num=filter_num, channel_num=channel_num)
        self.decoder.apply(weight_init)

        self.discrim = Discriminator(z_dim)
        self.discrim.apply(weight_init)

        # Optimizers
        self.optim_enc = optim.Adam(self.encoder.parameters(), lr=lr)
        self.optim_dec = optim.Adam(self.decoder.parameters(), lr=lr)
        self.optim_dis = optim.Adam(self.discrim.parameters(), lr=lr)

    def reset_gradients_(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.discrim.zero_grad()

    def train(self, loader, current_epoch):
        for idx, features in enumerate(tqdm(loader)):
            features = features[0]
            features = Variable(features)

            """ Decoding Phase """
            z_sample = self.encoder(features)
            features_sample = self.decoder(z_sample)

            dec_loss = F.binary_cross_entropy(
                features_sample, features
            )
            dec_loss.backward()
            self.optim_enc.step()
            self.optim_dec.step()
            self.reset_gradients_()

            """ Regularization phase """
            # Discriminator
            z_fake = self.encoder(features)
            z_real = Variable(torch.randn(
                features.size(0), self.encoder.z_dim))

            discrim_fake = self.discrim(z_fake)
            discrim_real = self.discrim(z_real)

            discrim_loss = - \
                torch.mean(torch.log(discrim_real) +
                           torch.log(1 - discrim_fake))
            discrim_loss.backward()

            self.optim_dis.step()
            self.reset_gradients_()

            # Encoder
            z_fake = self.encoder(features)
            discrim_fake = self.discrim(z_fake)

            enc_loss = -torch.mean(torch.log(discrim_fake))
            enc_loss.backward()

            self.optim_enc.step()
            self.reset_gradients_()

    def save_(self, e, filename='gan.path.tar'):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'discrim': self.discrim.state_dict(),
            'epoch': e + 1
        }, 'epoch{}_{}'.format(e, filename))

    def load_(self, filedir):
        if os.path.isfile(filedir):
            checkpoint = torch.load(filedir)

            self.encoder.load_state_dict(checkpoint['encoder'])
            self.decoder.load_state_dict(checkpoint['decoder'])
            self.discrim.load_state_dict(checkpoint['discrim'])

        else:
            print('Cant find file')
