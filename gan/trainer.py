import random
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image

from models import Generator, Encoder, Decoder, Discriminator, weight_init


def log(x):
    """ Log func to prevent nans """
    return torch.log(x + 1e-8)


class gan_trainer:
    def __init__(self, z_dim=32, h_dim=128, filter_num=64, channel_num=3,
                 lr=1e-3, cuda=False):
        # Are we cuda'ing it
        self.cuda = cuda

        # Generators
        # G(X) -> Y :: Image -> Smoothen
        # G(Y) -> X :: Smoothen -> Image
        self.g_xy = self.cudafy_(
            Generator(z_dim, h_dim=h_dim, filter_num=filter_num,
                    channel_num=channel_num)
        )
        self.g_xy.apply(weight_init)

        self.g_yx = self.cudafy_(
            Generator(z_dim, filter_num=filter_num, channel_num=channel_num)
        )
        self.g_yx.apply(weight_init)

        self.d_x = self.cudafy_(Discriminator(z_dim, filter_num, channel_num))
        self.d_x.apply(weight_init)

        self.d_y = self.cudafy_(Discriminator(z_dim, filter_num, channel_num))
        self.d_y.apply(weight_init)

        # Optimizers
        self.g_params = list(self.g_xy.parameters()) + list(self.g_yx.parameters())
        self.d_params = list(self.d_x.parameters()) + list(self.d_y.parameters())

        self.g_solver = optim.SGD(self.g_params, lr=lr, momentum=0.25)
        self.d_solver = optim.SGD(self.d_params, lr=lr, momentum=0.25)

        self.start_epoch = 0

    def cudafy_(self, m):
        if self.cuda:
            if hasattr(m, 'cuda'):
                return m.cuda()
        return m

    def reset_gradients_(self):
        self.g_xy.zero_grad()
        self.g_yx.zero_grad()
        self.d_x.zero_grad()
        self.d_y.zero_grad()

    def train(self, loader, current_epoch):
        for idx, features_ in enumerate(tqdm(loader)):
            features = features_[0]
            features = Variable(self.cudafy_(features))

            smoothen_features = features_[1]
            smoothen_features = Variable(self.cudafy_(smoothen_features))

            # Generators
            # G(X) -> Y :: Image -> Smoothen
            # G(Y) -> X :: Smoothen -> Image
            # Features X = features
            # Features Y = Smoothen

            # Discriminator Y
            gen_features_y = self.g_xy(features)
            discrim_y_real = self.d_y(smoothen_features)
            discrim_y_fake = self.d_y(gen_features_y)

            loss_d_y = -torch.mean(log(discrim_y_real) + log(1 - discrim_y_fake))

            # Discriminator X
            gen_features_x = self.g_yx(smoothen_features)
            discrim_x_real = self.d_x(features)
            discrim_x_fake = self.d_x(gen_features_x)

            loss_d_x = -torch.mean(log(discrim_x_real) + log(1 - discrim_x_fake))

            # Total discriminator loss
            discrim_loss = loss_d_y + loss_d_x
            discrim_loss.backward()
            self.d_solver.step()
            self.reset_gradients_()

            # Generator(X) -> Y
            # G(X) -> Y :: Image -> Smoothen
            gen_features_xy = self.g_xy(features)
            discrim_y_fake = self.d_y(gen_features_xy)
            gen_features_xyx = self.g_yx(gen_features_xy)

            loss_adv_y = -torch.mean(log(discrim_y_fake))
            loss_recons_x = torch.mean(torch.sum((features - gen_features_xyx) ** 2, 1))
            loss_g_xy = loss_adv_y + loss_recons_x

            # Generator(Y) -> X
            # G(Y) -> X :: Smoothen -> Image
            gen_features_yx = self.g_yx(smoothen_features)
            discrim_x_fake = self.d_x(gen_features_yx)
            gen_features_yxy = self.g_xy(gen_features_yx)

            loss_adv_x = -torch.mean(log(discrim_x_fake))
            loss_recons_y = torch.mean(torch.sum((smoothen_features - gen_features_yxy) ** 2, 1))
            loss_g_yx = loss_adv_x + loss_recons_y

            # Total Generator loss
            gen_loss = loss_g_xy + loss_g_yx
            gen_loss.backward()
            self.g_solver.step()
            self.reset_gradients_()

            tqdm.write(
                "Epoch: {}\t"
                "D loss: {:.4f}\t"
                "G loss: {:.4f}"
                .format(
                    current_epoch, discrim_loss.data[0], gen_loss.data[0]
                )
            )

        # Gets a random image and encode it to
        # get the latent space
        self.visualize(features, current_epoch)

    def reconstruct(self, img, transformers=None):
        if transformers is not None:
            img = transformers(img)

        img = Variable(self.cudafy_(img))
        img = img.view(1, img.size(0), img.size(1), img.size(2))

        return self.g_yx(img)

    def visualize(self, smoothen, e):
        """
        Visualize the training progress, for sanity checks

        Args:
            smoothen: smoothen image
            e: current epoch
        """
        if not os.path.exists('visualize/'):
            os.makedirs('visualize/')

        # Random image from sample
        idx = random.randint(0, smoothen.size(0) - 1)

        # Takes z sample and converts it to range
        # 0-255
        decoded = self.g_yx(smoothen)

        if self.cuda:
            decoded = decoded.data.cpu().numpy()
            smoothen = smoothen.data.cpu().numpy()
        else:
            decoded = decoded.data.numpy()
            smoothen = smoothen.data.numpy()
        dimg = self.tensor2pil(decoded[idx])
        oimg = self.tensor2pil(smoothen[idx])

        fig, axarr = plt.subplots(2, sharex=True)

        axarr[0].imshow(oimg)
        axarr[0].set_title('original @ epoch: {}'.format(e))

        axarr[1].imshow(dimg)
        axarr[1].set_title('decoded @ epoch : {}'.format(e))

        fig.savefig('visualize/{}.png'.format(e))

    @staticmethod
    def tensor2pil(t):
        # Assuming t is between 0.0 - 1.0
        t = t * 255
        t = t.astype(np.uint8)
        t = np.rollaxis(t, 0, 3)
        return Image.fromarray(t, 'RGB')

    def save_(self, e, filename='gan.path.tar'):
        torch.save({
            'g_xy': self.g_xy.state_dict(),
            'g_yx': self.g_yx.state_dict(),
            'd_x': self.d_x.state_dict(),
            'd_y': self.d_y.state_dict(),
            'epoch': e + 1
        }, 'epoch{}_{}'.format(e, filename))
        print('Saved model state')

    def load_(self, filedir):
        if os.path.isfile(filedir):
            checkpoint = torch.load(filedir)

            self.g_xy.load_state_dict(checkpoint['g_xy'])
            self.g_yx.load_state_dict(checkpoint['g_yx'])
            self.d_x.load_state_dict(checkpoint['d_x'])
            self.d_y.load_state_dict(checkpoint['d_y'])
            self.start_epoch = checkpoint['epoch']

            print('Model state loaded')

        else:
            print('Cant find file')
