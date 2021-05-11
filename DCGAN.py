from config import global_cfg
from torch import nn
from tqdm import tqdm
from tools import *
from PIL import Image
import os


CONV_DIM = 128
VEC_Z_SIZE = 100
EPOCHS = 150
G_LR = 0.0002
D_LR = G_LR
LR_DECAY_EPOCH = 10
LR_DECAY_GAMMA = 0.9
'''
origin DCGAN with 4 convolutional layers
'''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=CONV_DIM,
                               kernel_size=5,
                               padding=2,
                               stride=2,
                               bias=False)

        self.conv2 = nn.Conv2d(in_channels=CONV_DIM,
                               out_channels=2 * CONV_DIM,
                               kernel_size=5,
                               padding=2,
                               stride=2,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=CONV_DIM * 2)

        self.conv3 = nn.Conv2d(in_channels=2 * CONV_DIM,
                               out_channels=4 * CONV_DIM,
                               kernel_size=5,
                               padding=2,
                               stride=2,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=CONV_DIM * 4)

        self.conv4 = nn.Conv2d(in_channels=4 * CONV_DIM,
                               out_channels=8 * CONV_DIM,
                               kernel_size=5,
                               padding=2,
                               stride=2,
                               bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=CONV_DIM * 8)

        self.fc1 = nn.Linear(in_features=4 * 4 * 8 * CONV_DIM,
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

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.fn(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(VEC_Z_SIZE, 4 * 4 * CONV_DIM * 8)
        self.conv1 = nn.ConvTranspose2d(in_channels=CONV_DIM * 8,
                                        out_channels=CONV_DIM * 4,
                                        kernel_size=5,
                                        stride=2,
                                        padding=2,
                                        output_padding=1,
                                        bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=CONV_DIM * 4)

        self.conv2 = nn.ConvTranspose2d(in_channels=CONV_DIM * 4,
                                        out_channels=CONV_DIM * 2,
                                        kernel_size=5,
                                        stride=2,
                                        padding=2,
                                        output_padding=1,
                                        bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=CONV_DIM * 2)

        self.conv3 = nn.ConvTranspose2d(in_channels=CONV_DIM * 2,
                                        out_channels=CONV_DIM,
                                        kernel_size=5,
                                        stride=2,
                                        padding=2,
                                        output_padding=1,
                                        bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=CONV_DIM)

        self.conv4 = nn.ConvTranspose2d(in_channels=CONV_DIM,
                                        out_channels=3,
                                        kernel_size=5,
                                        stride=2,
                                        padding=2,
                                        output_padding=1,
                                        bias=False)
        self.fn = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.size(0), -1, 4, 4)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.fn(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.fn(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.fn(x)

        x = self.conv4(x)
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
    def __init__(self):
        self.D = Discriminator()
        self.G = Generator()
        self.D.apply(weights_init_normal)
        self.G.apply(weights_init_normal)
        self.gpu_available = torch.cuda.is_available()
        self.generated_train_images = './generated_images/train/Origin-DCGAN/'
        self.log_path = os.path.join(self.generated_train_images, 'log.txt')
        self.combined_image_folder = os.path.join(self.generated_train_images, 'combined_images')
        os.makedirs(self.generated_train_images, exist_ok=True)
        os.makedirs(self.combined_image_folder, exist_ok=True)
        with open(self.log_path, 'w') as f:
            pass
        if self.gpu_available:
            self.D.cuda()
            self.G.cuda()
        self.d_optim = torch.optim.Adam(self.D.parameters(), lr=D_LR)
        self.g_optim = torch.optim.Adam(self.G.parameters(), lr=G_LR)
        self.d_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.d_optim,
                                                              LR_DECAY_EPOCH,
                                                              LR_DECAY_GAMMA)
        self.g_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.g_optim,
                                                              LR_DECAY_EPOCH,
                                                              LR_DECAY_GAMMA)
        print(self.D)
        print(self.G)

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
                np.random.uniform(-1, 1, size=(global_cfg.batch_size, VEC_Z_SIZE))).float()
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
                np.random.uniform(-1, 1, size=(global_cfg.batch_size, VEC_Z_SIZE))).float()
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
        images = self.G(fixed_z)
        D_result = self.D(images).to('cpu').detach().numpy().copy()
        images = images.to('cpu').detach().numpy().copy()
        images = np.transpose(images, (0, 2, 3, 1))
        images = (images * 255).astype(np.uint8)
        folder = os.path.join(self.generated_train_images, str(current_epoch))
        os.makedirs(folder, exist_ok=True)
        for idx, image in enumerate(images):
            image = Image.fromarray(image)
            if D_result[idx] >= 0:
                image.save(os.path.join(folder, str(idx + 1) + '_positive.jpg'))
            else:
                image.save(os.path.join(folder, str(idx + 1) + '_negative.jpg'))
            image.close()
        display_images_in_folder(folder, os.path.join(self.combined_image_folder, '{}.jpg'.format(current_epoch)))

    def train(self, dataloader):
        fixed_z = torch.from_numpy(np.random.uniform(-1, 1, size=(global_cfg.batch_size, VEC_Z_SIZE))).float()
        fixed_z_np = fixed_z.numpy()
        print('fix z require grad = {}'.format(fixed_z.requires_grad))
        if self.gpu_available:
            fixed_z = fixed_z.cuda()
        for epoch in range(1, EPOCHS + 1):
            self.train_one_epoch(epoch, dataloader)
            self.generate_epoch(epoch, fixed_z)
            self.d_lr_scheduler.step()
            self.g_lr_scheduler.step()
            fixed_z_trained = fixed_z.to('cpu').numpy()
            if np.array_equal(fixed_z_np, fixed_z_trained):
                print('fixed z changed!')
            else:
                print('fixed_z not changed')
            torch.save(self.G, os.path.join(self.generated_train_images, 'Generator.pt'))
            torch.save(self.D, os.path.join(self.generated_train_images, 'Discriminator.pt'))

    def forward(self, x):
        pass
