from torch.utils.data import DataLoader
import torchvision
from config import global_cfg
import os

def createDataLoader():
    '''
    generate a dataloader from global_cfg.data_path.
    The images are normalized to [0, 1]
    :return:
    '''
    for root, _, files in os.walk(global_cfg.data_path):
        for file in files:
            file_size = os.path.getsize(os.path.join(root, file))
            if file_size == 0:
                print('file {} is empty. Removed'.format(file))
                os.remove(os.path.join(root, file))
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((global_cfg.image_size, global_cfg.image_size)),
            torchvision.transforms.ToTensor()
        ]
    )

    dataset = torchvision.datasets.ImageFolder(global_cfg.data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=global_cfg.batch_size, shuffle=True)
    return dataloader