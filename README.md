This repository implements the 4-layer DCGAN with Pytorch and trained with [Anime-Face-Dataset](https://github.com/bchao1/Anime-Face-Dataset).
The generated images is 64 x 64.

# Set up environment
use Anaconda to create the environment from environment.yml

# To train with your own dataset:
modify "global_cfg.data_path" to your image folder from config.py
```
python train.py
```
# generate Anime Faces
```
python inference.py --num_images 128
```
