from torch.utils.data import DataLoader
import torchvision
from config import global_cfg


def createDataLoader():
    '''
    generate a dataloader from global_cfg.data_path.
    The images are normalized to [0, 1]
    :return:
    '''
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((global_cfg.image_size, global_cfg.image_size)),
            torchvision.transforms.ToTensor()
        ]
    )

    dataset = torchvision.datasets.ImageFolder(global_cfg.data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=global_cfg.batch_size, shuffle=True)
    return dataloader