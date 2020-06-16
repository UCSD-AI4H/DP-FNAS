# The source code of FNAS & DP-FNAS

Differentially-private Federated Neural Architecture Search

## Requirement

Slurm Platform with GPU support
Python 3.6
```
numpy==1.18.1
graphviz==2.40.1
python-graphviz==0.14
torch==1.5.0
torchvision==0.6.0
tensorboard==2.2.1
tensorboardX==2.0
```
Note: First install graphviz library using apt install and then pip install python-graphviz.


## Usage for FNAS
Adjust the batch size if out of memory (OOM) occurs. It dependes on your gpu memory size and genotype.

1. Search:

```shell
cd ./cnn
srun --nodes=1 --gres=gpu:1 python ./search.py --name cifar10 --dataset cifar10 --batch_size 256 --workers 16 --print_freq 10 --w_lr 0.05 --w_lr_min 0.002 --alpha_lr 0.0006 --infi_band False
srun --nodes=2 --gres=gpu:1 python ./search.py --name cifar10 --dataset cifar10 --batch_size 256 --workers 16 --print_freq 10 --w_lr 0.05 --w_lr_min 0.002 --alpha_lr 0.0006 --infi_band False
srun --nodes=4 --gres=gpu:1 python ./search.py --name cifar10 --dataset cifar10 --batch_size 256 --workers 16 --print_freq 10 --w_lr 0.05 --w_lr_min 0.002 --alpha_lr 0.0006 --infi_band False
srun --nodes=8 --gres=gpu:1 python ./search.py --name cifar10 --dataset cifar10 --batch_size 256 --workers 16 --print_freq 10 --w_lr 0.05 --w_lr_min 0.002 --alpha_lr 0.0006 --infi_band False
```
```shell
cd ./rnn
srun --nodes=1 --gres=gpu:1 python ./train_dist_search.py --name PTB --infi_band False --workers 16 --log-interval 10 --epochs 160 --small_batch_size 64
srun --nodes=2 --gres=gpu:1 python ./train_dist_search.py --name PTB --infi_band False --workers 16 --log-interval 10 --epochs 160 --small_batch_size 64
srun --nodes=4 --gres=gpu:1 python ./train_dist_search.py --name PTB --infi_band False --workers 16 --log-interval 10 --epochs 160 --small_batch_size 64
srun --nodes=8 --gres=gpu:1 python ./train_dist_search.py --name PTB --infi_band False --workers 16 --log-interval 10 --epochs 160 --small_batch_size 64
```

2. Augment:

```shell
cd ./cnn
./augment.py --nodes=1 --gres=gpu:1 --name cifar10 --dataset cifar10 --workers 16 --print_freq 50 --lr 0.05 --infi_band False --genotype {genotype_str}
./augment.py --nodes=2 --gres=gpu:1 --name cifar10 --dataset cifar10 --workers 16 --print_freq 50 --lr 0.05 --infi_band False --genotype {genotype_str}
./augment.py --nodes=4 --gres=gpu:1 --name cifar10 --dataset cifar10 --workers 16 --print_freq 50 --lr 0.05 --infi_band False --genotype {genotype_str}
./augment.py --nodes=8 --gres=gpu:1 --name cifar10 --dataset cifar10 --workers 16 --print_freq 50 --lr 0.05 --infi_band False --genotype {genotype_str}
```

```shell
cd ./rnn
srun python --nodes=1 --gres=gpu:1 ./train_dist.py --name PTB --infi_band False --workers 16 --log-interval 10 --small_batch_size 256  --batch_size 256
srun python --nodes=2 --gres=gpu:1 ./train_dist.py --name PTB --infi_band False --workers 16 --log-interval 10 --small_batch_size 256  --batch_size 256
srun python --nodes=4 --gres=gpu:1 ./train_dist.py --name PTB --infi_band False --workers 16 --log-interval 10 --small_batch_size 128  --batch_size 256
srun python --nodes=8 --gres=gpu:1 ./train_dist.py --name PTB --infi_band False --workers 16 --log-interval 10 --small_batch_size 64  --batch_size 256
```

## Usage for DP-FNAS

1. Search:
Add the following parameters to the shell scripts:
```shell
--dist_privacy True --var_sigma 1.0 --var_gamma 1.0 --max_hessian_grad_norm 0.1 --max_weights_grad_norm 0.01
```

2. Augment:
Add the following parameters to the shell scripts:
```shell
--dist_privacy True --var_sigma 1.0 --var_gamma 1.0 --max_hessian_grad_norm 0.1 --max_weights_grad_norm 0.01
```

## Results

The result for architecure search is in the `searchs` folder and augmentation is in the `augments` folder. Each individual GPU will generates its own model and logs.

## Data

The PTB dataset is provided in the `data` folder and the cifar10 dataset will be downloaded automatically.
