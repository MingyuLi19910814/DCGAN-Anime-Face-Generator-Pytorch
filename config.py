from easydict import EasyDict as edict


global_cfg = edict()

global_cfg.data_path = '../train-images/'
global_cfg.image_size = 64
global_cfg.init_lr = 0.002
global_cfg.lr_decay_epoch = 10
global_cfg.lr_decay_gamma = 0.8
global_cfg.train_epochs = 100
global_cfg.batch_size = 128
global_cfg.conv_dim = 128
global_cfg.img_mean = [0.485, 0.456, 0.406]
global_cfg.img_std = [0.229, 0.224, 0.225]
