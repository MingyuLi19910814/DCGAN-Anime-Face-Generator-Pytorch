This repository implements the 4-layer DCGAN with Pytorch and trained with [Anime-Face-Dataset](https://github.com/bchao1/Anime-Face-Dataset).
The generated images is 64 x 64.

# Set up environment
use Anaconda to create the environment from environment.yml

# Train with your own dataset:
modify "global_cfg.data_path" to your image folder from config.py
```
python train.py
```
# Generate Anime Faces
```
python inference.py --num_images 128
```
# demo
![github](https://github.com/MingyuLi19910814/DCGAN-Anime-Face-Generator-Pytorch/tree/master/demo.gif)
