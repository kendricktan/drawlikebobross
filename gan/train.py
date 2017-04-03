import argparse
import os
import torch
import torchvision.transforms as transforms

from .trainer import gan_trainer
from .loader import BobRossDataset

# Params
parser = argparse.ArgumentParser(description='GAN trainer')
parser.add_argument('--epoch', default=500, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--cuda', default='true', type=str)
parser.add_argument('--resume', default='', type=str)
args, unknown = parser.parse_known_args()

cuda = True if 'true' in args.cuda.lower() else False
cuda = True

transformers = transforms.Compose([
    transforms.ToTensor(),
])

# Gan trainer
trainer = gan_trainer(z_dim=32, h_dim=128, filter_num=64, channel_num=3, lr=args.lr, cuda=cuda)

if __name__ == '__main__':
    if args.resume:
        trainer.load_(args.resume)

    # dataset
    train_dataset = BobRossDataset('../images', transform=transformers)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=8, shuffle=True,
        pin_memory=cuda, num_workers=4
    )

    for e in range(trainer.start_epoch, args.epoch):
        trainer.train(train_loader, e)
        trainer.save_(e)
