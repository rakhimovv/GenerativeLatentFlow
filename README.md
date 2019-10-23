# Generative Latent Flow

This is an unofficial PyTorch implementation of the [Generative Latent Flow
](https://arxiv.org/abs/1905.10485) paper by Zhisheng Xiao, Qing Yan, Yali Amit.


## Requirements

The repository is built on top of PyTorch `1.2.0`. 
The structure of the project is inspired by [open-mmlab/mmsr](https://github.com/open-mmlab/mmsr).
To install all required packages, run the following command:

```bash
pip install -r requirements.txt
```

## How to use

The **Generative Latent Flow** (GLF) is an algorithm for generative modeling of the data distribution. 
One could use it to generate images. 

### Training

To start the training process run

```bash
python train.py -opt glf/options/train/train_glf_original_mnist.yml
```

To start a `TensorBoard` simply run

```bash
tensorboard --logdir tb_logger/experiment_name
```

### Parameters


If you would to modify models' hyperparameters, number of epochs, batch size, and other, 
consider looking to `glf/options/train` folder for configuration examples.
An example of the config file is as follows:

```yaml
name: my_experiment                     # name of the experiment
use_tb_logger: true                     # use tensorboard or not
model: generative
gpu_ids: [1,2]                          # GPUs to use, # e.g. [0,1,2,5]

datasets:
  train:
    name: MNIST                         # See `datasets` section for available datasets
    dataroot: datasets/mnist            # Path to the folder where data will be saved
    use_shuffle: true
    n_workers: 8                        # Num workers per GPU
    batch_size: 256
  val:
    name: MNIST
    dataroot: datasets/mnist

encoder: # (E)
  which_model: glf_encoder
  img_size: 28                          # H and W of the image
  in_ch: 1                              # Number of channels in the input image
  nz: 20                                # Dimension of the latent space

decoder: # (D)
  which_model: glf_decoder
  img_size: 28
  out_ch: 1
  nz: 20        

flow: # (F)
  which_model: FlowNet
  stop_gradients: True                  # Enable stop gradient operation (see the paper for more details)
  nz: 20
  hidden_size: 64
  nblocks: 4

path:
  pretrained_encoder: ~
  pretrained_decoder: ~
  pretrained_flow: ~
  resume_state: ~
  strict_load: true

train:
  lr_E: 0.001                           # 0 to freeze params
  weight_decay_E: 0
  beta1_E: 0.9
  beta2_E: 0.99

  lr_D: 0.001                           # 0 to freeze params
  weight_decay_D: 0
  beta1_D: 0.9
  beta2_D: 0.99

  lr_F: 0.001                           # 0 to freeze params
  weight_decay_F: 0.00002
  beta1_F: 0.9
  beta2_F: 0.99

  niter: 23400                          # Number of iterations

  lr_scheme: MultiStepLR
  warmup_iter: -1                       # -1 means no warm up
  lr_steps: [11700]
  lr_gamma: 0.5

  pixel_criterion: l1                   # Reconstruction loss (`l1` or `l2`)
  pixel_weight: 1                       # Weight of the reconstruction loss in the final loss
  add_background_mask: false            # Will put higher weight to regions where (image != 1). Useful for dots dataset

  feature_weight: 0                     # Weight of the perceptual loss in the final loss

  nll_weight: 1                         # Weight of the NLL loss in the final loss

  manual_seed: 10
  val_calculate_fid_prd: true           # Calculate FID and PRD during training
  val_freq: 1000                        # How often to start the validation process

logger:
  print_freq: 100
  save_checkpoint_freq: 10000
```

There are some prepared configs in `glf/options/train` folder.


This command will create multiple folders

```
tb_logger/experiment_name                   # Tensorboard event files
└──  events.out.tfevents
experiments/experiment_name
├── models                                  # Saved checkpoints
│   ├── latest_D.pth
│   ├── latest_F.pth
│   └── latest_G.pth
├── samples                                 # Examples of generated images
│   ├── 1000.png
│   ├── 2000.png
│   ├── ...
│   └── 100500.png
└──  config.ymllogger_file.log              # Logs
```

### Datasets

The following datasets are available: [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html),
MNIST, FashionMNIST, CIFAR10, [Dots](https://arxiv.org/abs/1811.03259). 
If you would like to add new dataset, please look at `glf/data/__init__.py` for an example.


### Noise interpolation

To play with noise interpolation and image generation check the `notebooks/celeba_demo.ipynb`.

<p align="center"><img src="imgs/noise_example.png" width="480"\></p>


## Results

### CelebA

This result is achieved by using the `glf/options/train/train_glf_original_celeba.yml` config.


<p align="center"><img src="imgs/celeba_example.jpg" width="480"\></p>


### CIFAR10

This result is achieved by using the `glf/options/train/train_glf_original_cifar10.yml` config.


<p align="center"><img src="imgs/cifar10_example.jpg" width="480"\></p>