import numpy as np
import torch
from torch import nn
from config import global_cfg
from tqdm import tqdm
from tools import *
from PIL import Image
import os

'''
Naive DCGAN with 3 convolutional layers
'''
class Discriminator(nn.Module):
    def __init__(self, conv_dim):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=conv_dim,
                               kernel_size=4,
                               padding=1,
                               stride=2,
                               bias=False)

        self.conv2 = nn.Conv2d(in_channels=conv_dim,
                               out_channels=2 * conv_dim,
                               kernel_size=4,
                               padding=1,
                               stride=2,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=conv_dim * 2)

        self.conv3 = nn.Conv2d(in_channels=2 * conv_dim,
                               out_channels=4 * conv_dim,
                               kernel_size=4,
                               padding=1,
                               stride=2,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=conv_dim * 4)

        resized_img_size = global_cfg.image_size // 8

        self.fc1 = nn.Linear(in_features=resized_img_size * resized_img_size * 4 * conv_dim,
                             out_features=1)
        self.fn = nn.LeakyReLU(0.2)
    '''
    inference the image.
    The input should be within range [0, 1]
    '''
    def forward(self, x):
        x = self.conv1(x)
        x = self.fn(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.fn(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.fn(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class Generator(nn.Module):
    def __init__(self, random_vector_size, conv_dim):
        super(Generator, self).__init__()
        self.resized_img_size = global_cfg.image_size // 8
        self.fc1 = nn.Linear(random_vector_size, self.resized_img_size * self.resized_img_size * 4 * conv_dim)
        self.conv1 = nn.ConvTranspose2d(in_channels=conv_dim * 4,
                                        out_channels=conv_dim * 2,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=conv_dim * 2)

        self.conv2 = nn.ConvTranspose2d(in_channels=conv_dim * 2,
                                        out_channels=conv_dim,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=conv_dim)

        self.conv3 = nn.ConvTranspose2d(in_channels=conv_dim,
                                        out_channels=3,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias=False)
        # self.fn = nn.ReLU()
        self.fn = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.size(0), -1, self.resized_img_size, self.resized_img_size)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.fn(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.fn(x)

        x = self.conv3(x)

        x = torch.nn.functional.tanh(x)
        # the output of tanh is [-1, 1]
        # ensure the range is [0, 1]
        x = (x + 1) / 2
        return x

def real_loss(D_out):
    device = D_out.device
    labels = torch.ones(D_out.size(0)) * 0.9
    labels = labels.to(device)
    loss = nn.BCEWithLogitsLoss()(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    device = D_out.device
    labels = torch.zeros(D_out.size(0))
    labels = labels.to(device)
    loss = nn.BCEWithLogitsLoss()(D_out.squeeze(), labels)
    return loss


class Model:
    def __init__(self, conv_dim, random_vector_size):
        # normalize the real image and fake images
        self.conv_dim = conv_dim
        self.random_vector_size = random_vector_size
        self.D = Discriminator(conv_dim)
        self.G = Generator(random_vector_size, conv_dim)
        self.D.apply(weights_init_normal)
        self.G.apply(weights_init_normal)
        self.gpu_available = torch.cuda.is_available()
        self.generated_train_images = './generated_images/train/NaiveDCGAN3/'
        self.log_path = os.path.join(self.generated_train_images, 'log.txt')
        os.makedirs(self.generated_train_images, exist_ok=True)
        with open(self.log_path, 'w') as f:
            pass
        if self.gpu_available:
            self.D.cuda()
            self.G.cuda()
        self.d_optim = torch.optim.Adam(self.D.parameters(), lr=global_cfg.init_lr / 4)
        self.g_optim = torch.optim.Adam(self.G.parameters(), lr=global_cfg.init_lr)
        self.d_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.d_optim,
                                                              global_cfg.lr_decay_epoch,
                                                              global_cfg.lr_decay_gamma)
        self.g_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.g_optim,
                                                              global_cfg.lr_decay_epoch,
                                                              global_cfg.lr_decay_gamma)

    def normalize_images(self, tensor, mean, std):
        '''

        :param images: tensor (B, C, H, W)
        :return:
        '''
        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        return tensor

    def train_one_epoch(self, current_epoch, dataloader):
        self.D.train()
        self.G.train()
        d_loss_vec = []
        g_loss_vec = []
        bar = tqdm(total=dataloader.__len__())
        for step, (images, _) in enumerate(dataloader):
            '''
            train discriminator
            '''
            if self.gpu_available:
                images = images.cuda()
            real_images = self.normalize_images(images, global_cfg.img_mean, global_cfg.img_std)
            self.d_optim.zero_grad()
            # train on real images
            real_images_out = self.D(real_images)
            real_images_loss = real_loss(real_images_out)
            # generate fake images
            z = torch.from_numpy(
                np.random.uniform(-1, 1, size=(global_cfg.batch_size, self.random_vector_size))).float()
            if self.gpu_available:
                z = z.cuda()
            fake_images = self.G(z)
            fake_images = self.normalize_images(fake_images, global_cfg.img_mean, global_cfg.img_std)
            fake_images_out = self.D(fake_images)
            fake_images_loss = fake_loss(fake_images_out)

            d_loss = real_images_loss + fake_images_loss
            d_loss.backward()
            self.d_optim.step()
            '''
            train generator
            '''
            self.g_optim.zero_grad()
            z = torch.from_numpy(
                np.random.uniform(-1, 1, size=(global_cfg.batch_size, self.random_vector_size))).float()
            if self.gpu_available:
                z = z.cuda()
            fake_images = self.G(z)
            fake_images = self.normalize_images(fake_images, global_cfg.img_mean, global_cfg.img_std)
            fake_images_out = self.D(fake_images)
            g_loss = real_loss(fake_images_out)
            g_loss.backward()
            self.g_optim.step()
            bar.update(1)
            d_loss_vec.append(d_loss.item())
            g_loss_vec.append(g_loss.item())
        bar.close()
        msg = 'epoch = {}, d_loss = {}, g_loss = {}, d_lr = {}, g_lr = {}\n'.format(current_epoch,
                                                                                  np.average(d_loss_vec),
                                                                                  np.average(g_loss_vec),
                                                                                  self.get_current_lr(self.d_optim),
                                                                                  self.get_current_lr(self.g_optim))
        with open(self.log_path, 'a') as f:
            f.write(msg)
        print(msg)

    def get_current_lr(self, optim):
        return optim.state_dict()['param_groups'][0]['lr']

    def generate_epoch(self, current_epoch, fixed_z):
        self.G.eval()
        images = self.G(fixed_z).to('cpu').detach().numpy().copy()
        images = np.transpose(images, (0, 2, 3, 1))
        images = (images * 255).astype(np.uint8)
        folder = os.path.join(self.generated_train_images, str(current_epoch))
        os.makedirs(folder, exist_ok=True)
        for idx, image in enumerate(images):
            image = Image.fromarray(image)
            image.save(os.path.join(folder, str(idx + 1) + '.jpg'))
            image.close()

    def train(self, dataloader):
        fixed_z = torch.from_numpy(np.random.uniform(-1, 1, size=(global_cfg.batch_size, self.random_vector_size))).float()
        if self.gpu_available:
            fixed_z = fixed_z.cuda()
        for epoch in range(1, global_cfg.train_epochs):
            self.train_one_epoch(epoch, dataloader)
            self.generate_epoch(epoch, fixed_z)
            self.d_lr_scheduler.step()
            self.g_lr_scheduler.step()
            torch.save(self.G, os.path.join(self.generated_train_images, 'Generator.pt'))
            torch.save(self.D, os.path.join(self.generated_train_images, 'Discriminator.pt'))

    def forward(self, x):
        pass
