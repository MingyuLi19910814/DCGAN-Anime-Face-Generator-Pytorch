import torch
import numpy as np
import os
import argparse
from PIL import Image
from tools import display_images_in_folder

parser = argparse.ArgumentParser('Generated Anime Faces')
parser.add_argument('--num_images', type=int, default=64, help='number of generated images')
args = parser.parse_args()

if __name__ == "__main__":
    generated_image_folder = './result'
    os.makedirs(generated_image_folder, exist_ok=True)
    G = torch.load('./model/Generator.pt')
    num_images = args.num_images

    fixed_z = torch.from_numpy(np.random.uniform(-1, 1, size=(num_images, 100))).float()
    if torch.cuda.is_available():
        fixed_z = fixed_z.cuda()
        G.cuda()
    images = G(fixed_z)
    images = images.to('cpu').detach().numpy().copy()
    images = np.transpose(images, (0, 2, 3, 1))
    images = (images * 255).astype(np.uint8)

    for idx, image in enumerate(images):
        file_name = os.path.join(generated_image_folder, '{}.jpg'.format(idx + 1))
        Image.fromarray(image).save(file_name)
    display_images_in_folder(generated_image_folder, os.path.join(generated_image_folder, 'combined.jpg'))
