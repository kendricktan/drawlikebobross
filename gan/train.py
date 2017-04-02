import os
import torch
import torchvision.transforms as transforms

from trainer import gan_trainer
from loader import BobRossDataset

transformers = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = BobRossDataset('../images', transform=transformers)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=12, shuffle=True,
)

trainer = gan_trainer(z_dim=32, h_dim=128, filter_num=64, channel_num=3, lr=1e-3)

for e in range(500):
    trainer.train(train_loader, e)
