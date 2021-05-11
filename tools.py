import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import imageio
from tqdm import tqdm

def weights_init_normal(layer, mean = 0.0, std = 0.02):
    layer_name = layer.__class__.__name__
    if layer_name in ['Conv2d', 'Linear']:
        layer.weight.data.normal_(mean, std)

def display_images_in_folder(path, save_path='None', display=False):
    for root, directory, files in os.walk(path):
        files = [file for file in files if os.path.splitext(file)[-1] == '.jpg']
        images_np = []
        for file in files:
            file = os.path.join(root, file)
            with Image.open(file) as im:
                image_np = np.asarray(im)
                images_np.append(image_np)
        col = int(np.sqrt(images_np.__len__()))
        row = (images_np.__len__() - 1) // col + 1
        fig = plt.figure(figsize=(row, col))

        for rr in range(1, row + 1):
            for cc in range(1, col + 1):
                idx = (rr - 1) * col + cc - 1
                if idx < images_np.__len__():
                    fig.add_subplot(row, col, idx + 1)
                    f = plt.imshow(images_np[idx])
                    f.axes.get_xaxis().set_visible(False)
                    f.axes.get_yaxis().set_visible(False)
        if save_path is not None:
            plt.savefig(save_path)
        if display:
            plt.show()

def generate_video(combined_image_folder, fps=10):
    assert os.path.isdir(combined_image_folder)

    for root, directory, files in os.walk(combined_image_folder):
        files = [file for file in files if os.path.splitext(file)[-1] == '.jpg']
        files = sorted(files, key=lambda file: int(file[:-4]))
        files = [os.path.join(root, file) for file in files]
        assert files.__len__() > 0
        im = cv2.imread(files[0])
        height, width = im.shape[:2]

        video_output_path = os.path.join(combined_image_folder, 'video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_output = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
        bar = tqdm(total=files.__len__())
        for file in files:
            im = cv2.imread(file)
            video_output.write(im)
            bar.update(1)
        bar.close()

def generate_gif(combined_image_folder, duration=0.5, sample=3):
    assert os.path.isdir(combined_image_folder)

    for root, directory, files in os.walk(combined_image_folder):
        files = [file for file in files if os.path.splitext(file)[-1] == '.jpg']
        files = sorted(files, key=lambda file: int(file[:-4]))
        files = [os.path.join(root, file) for file in files]
        assert files.__len__() > 0

        images = []
        for idx, filename in enumerate(files):
            if idx % sample == 0:
                images.append(imageio.imread(filename))
        imageio.mimsave(os.path.join(combined_image_folder, 'demo.gif'), images, duration=duration)