# Action Recognition with metric learning
This repo is to verify if metric learning method is useful in action recongnition with 3DCNN.

The result is coming soon.

## Requirements
* python 3.x
* pytorch >= 1.0
* torchvision
* pandas
* numpy
* Pillow
* tqdm
* PyYAML
* addict
* tensorboardX
* adabound
* (accimage)

## Dataset
### Kinetics

You can download videos in Kinetics with [the official donwloader](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).

Then you need to convert .mp4 files to .jpeg files.
You can do that using the code from [this repository](https://github.com/kenshohara/3D-ResNets-PyTorch/tree/work).

## Training
If you want to train a model, please run `python utils/build_dataset.py` to make csv_files for training and validation.

Then, just run `python train.py ./PATH_TO_CONFIG_FILE`

For example, when running `python train.py ./result/resnet18/adacos/config.yaml`,
the configuration described in `./result/resnet18/adacos/config.yaml` will be used .

If you want to set your own configuration, please make config.yaml like this:
```
model: resnet18
metric: None

class_weight: True    # if you use class weight to calculate cross entropy or not
writer_flag: True      # if you use tensorboardx or not

n_classes: 400
batch_size: 32
input_frames: 16
height: 224
width: 224
num_workers: 8
max_epoch: 250

optimizer: SGD
learning_rate: 0.001
lr_patience: 10       # Patience of LR scheduler
momentum: 0.9         # momentum of SGD
dampening: 0.0        # dampening for momentum of SGD
weight_decay: 0.0001   # weight decay
nesterov: True        # enables Nesterov momentum
final_lr: 0.1         # final learning rate for AdaBound

dataset_dir: dataset_dir: /groups1/gaa50031/aaa10329ah/datasets/kinetics/videos_400_jpg
train_csv: ./dataset/kinetics_400_train.csv
val_csv: ./dataset/kinetics_400_val.csv
result_path: ./result/resnet18/fc
```

## Result
Coming Soon

## References
* [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)
* [adacos](https://github.com/4uiiurz1/pytorch-adacos)

# TODO
* add config as to jpg or hdvu or mp4