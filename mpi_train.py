import argparse
import os
import pandas as pd
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml

from addict import Dict
from math import ceil
from random import Random
from tensorboardX import SummaryWriter
from torch.multiprocessing import Process
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomCrop, Normalize

from utils.checkpoint import save_checkpoint, resume
from utils.class_weight import get_class_weight
from utils.dataset import Kinetics
from utils.mean import get_mean, get_std
from model import resnet


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='train a network for action recognition')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Add --resume option if you start training from checkpoint.'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        metavar='S', help='random seed(default: 42)'
    )

    return parser.parse_args()


# for distributed training
class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


# for distributed training
class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


# for distributed training
def partition_dataset(dataset, config):
    """ Partitioning dataset """
    size = dist.get_world_size()
    bsz = config.batch_size / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    partition_loader = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True, num_workers=config.num_workers, drop_last=True
    )
    return partition_loader, bsz


# for distributed training
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


def accuracy(output, target, topk=(1, )):
    """
    Computes the accuracy over the k top predictions
    """

    N = output.shape[0]
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return N, res


###############################TODO: below

def validation(model, val_loader, criterion, config, device):
    model.eval()
    val_loss = 0.0
    n_samples = 0.0
    top1 = 0.0
    top5 = 0.0

    with torch.no_grad():
        for sample in val_loader:
            # temporal size is input_frames(default 16) * interger
            x = sample['clip']
            x = x.to(device)
            t = sample['cls_id']
            t = t.to(device)

            h = model(x)

            val_loss += criterion(h, t).item()
            n, topk = accuracy(h, t, topk=(1, 5))
            n_samples += n
            top1 += topk[0].item()
            top5 += topk[1].item()

        val_loss /= len(val_loader)
        top1 /= n_samples
        top5 /= n_samples

    return val_loss, top1, top5


# for distributed training
def run(rank, size, model, optimizer, train_data, val_data, criterion, config, device):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    train_set, train_bsz = partition_dataset(train_data, config)
    val_set, val_bsz = partition_dataset(val_set, config)

    def train(model, epoch, train_loader, criterion, optimizer):
        model.train()

    epoch_loss = 0.0
    for sample in train_loader:
        x = sample['clip']
        t = sample['cls_id']
        x = x.cuda()
        t = t.cuda()

        h = model(x)
        loss = criterion(h, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
#            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.data[0]
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches)


def main():
    args = get_arguments()

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    # writer
    if CONFIG.writer_flag:
        writer = SummaryWriter(CONFIG.result_path)
    else:
        writer = None

    # DataLoaders
    normalize = Normalize(mean=get_mean(), std=get_std())

    train_data = Kinetics(
        CONFIG,
        transform=Compose([
            RandomCrop((CONFIG.height, CONFIG.width)),
            ToTensor(),
            normalize,
        ])
    )

    val_data = Kinetics(
        CONFIG,
        transform=Compose([
            RandomCrop((CONFIG.height, CONFIG.width)),
            ToTensor(),
            normalize,
        ]),
        mode='validation'
    )

    # load model
    print('\n------------------------Loading Model------------------------\n')

    if CONFIG.model == 'resnet18':
        print(CONFIG.model + ' will be used as a model.')
        model = resnet.generate_model(18, n_classes=CONFIG.n_classes)
    elif CONFIG.model == 'resnet50':
        print(CONFIG.model + ' will be used as a model.')
        model = resnet.generate_model(50, n_classes=CONFIG.n_classes)
    else:
        print('resnet18 will be used as a model.')
        model = resnet.generate_model(18, n_classes=CONFIG.n_classes)

    # set optimizer, lr_scheduler
    if CONFIG.optimizer == 'Adam':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG.learning_rate
        )
    elif CONFIG.optimizer == 'SGD':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.SGD(
            model.parameters(),
            lr=CONFIG.learning_rate,
            momentum=CONFIG.momentum,
            dampening=CONFIG.dampening,
            weight_decay=CONFIG.weight_decay,
            nesterov=CONFIG.nesterov
        )
    else:
        print('There is no optimizer which suits to your option. \
            Instead, SGD will be used as an optimizer.')
        optimizer = optim.SGD(
            model.parameters(),
            lr=CONFIG.learning_rate,
            momentum=CONFIG.momentum,
            dampening=CONFIG.dampening,
            weight_decay=CONFIG.weight_decay,
            nesterov=CONFIG.nesterov
        )

    # learning rate scheduler
    if CONFIG.optimizer == 'SGD':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=CONFIG.lr_patience)
    else:
        scheduler = None

    # resume if you want
    begin_epoch = 0
    log = None
    if args.resume:
        if os.path.exists(os.path.join(CONFIG.result_path, 'checkpoint.pth')):
            print('loading the checkpoint...')
            begin_epoch, model, optimizer, scheduler = resume(
                CONFIG, model, optimizer, scheduler)
            print('training will start from {} epoch'.format(begin_epoch))
            log = pd.read_csv(
                os.path.join(CONFIG.result_path, 'log.csv')
            )

    # generate log when you start training from scratch
    if log is None:
        log = pd.DataFrame(
            columns=['epoch', 'lr', 'train_loss', 'val_loss', 'acc@1', 'acc@5']
        )

    # criterion for loss
    if CONFIG.class_weight:
        criterion = nn.CrossEntropyLoss(weight=get_class_weight())
    else:
        criterion = nn.CrossEntropyLoss()

    # train and validate model
    print('\n------------------------Start training------------------------\n')
    train_losses = []
    val_losses = []
    top1_accuracy = []
    top5_accuracy = []
    best_top1_accuracy = 0.0

    for epoch in range(begin_epoch, CONFIG.max_epoch):

        start_time = time.time()

        # training
        train_loss = train(
            model, epoch, train_loader, train_sampler, criterion, optimizer, CONFIG)
        train_losses.append(train_loss)

        # validation
        val_loss, top1, top5 = validation(
            model, epoch, val_loader, val_sampler, criterion, CONFIG)

        if CONFIG.optimizer == 'SGD':
            scheduler.step(val_loss)

        val_losses.append(val_loss)
        top1_accuracy.append(top1)
        top5_accuracy.append(top5)

        # save checkpoint every epoch
        save_checkpoint(CONFIG, epoch, model, optimizer, scheduler)

        # save a model if top1 accuracy is higher than ever
        # save a base model, NOT DataParalled model
        if best_top1_accuracy < top1_accuracy[-1]:
            best_top1_accuracy = top1_accuracy[-1]
            torch.save(
                model.module.state_dict(),
                os.path.join(
                    CONFIG.result_path, 'best_top1_accuracy_model.prm'
                )
            )

        # save a model every 10 epoch
        if epoch % 10 == 0 and epoch != 0:
            torch.save(
                model.module.state_dict(),
                os.path.join(
                    CONFIG.result_path,
                    'epoch_{}_model.prm'.format(epoch)
                )
            )

        # tensorboardx
        if writer is not None:
            writer.add_scalar("train_loss", train_losses[-1], epoch)
            writer.add_scalar('val_loss', val_losses[-1], epoch)
            writer.add_scalars(
                "iou", {
                    'top1_accuracy': top1_accuracy[-1],
                    'top5_accuracy': top5_accuracy[-1]
                }, epoch)

        # write logs to dataframe and csv file
        tmp = pd.Series([
            epoch,
            scheduler.get_lr()[0],
            train_losses[-1],
            val_losses[-1],
            top1_accuracy[-1],
            top5_accuracy[-1],
        ], index=log.columns)

        log = log.append(tmp, ignore_index=True)
        log.to_csv(os.path.join(CONFIG.result_path, 'log.csv'), index=False)

        print(
            'elapsed time: {}\tepoch: {}\tloss train: {:.5f}\tloss val: {:.5f}\ttop1_accuracy: {:.5f}\ttop5_accuracy: {:.5f}'
            .format(time.time() - start_time, epoch, train_losses[-1], val_losses[-1], top1_accuracy[-1], top5_accuracy[-1])
        )

    torch.save(
        model.module.state_dict(),
        os.path.join(CONFIG.result_path, 'final_model.prm')
    )


if __name__ == '__main__':
    main()
