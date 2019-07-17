import argparse
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from addict import Dict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomCrop, Normalize

from utils.checkpoint import save_checkpoint, resume
from utils.class_weight import get_class_weight
from utils.dataset import Kinetics
from utils.mean import get_mean, get_std
from model import resnet

# for distributed training
import horovod.torch as hvd


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


def train(model, epoch, train_loader, train_sampler, criterion, optimizer, config, device):
    model.train()

    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)

    epoch_loss = 0.0
    for sample in train_loader:
        x = sample['clip']
        t = sample['cls_id']
        x = x.to(device)
        t = t.to(device)

        h = model(x)
        loss = criterion(h, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


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


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def validation(model, epoch, val_loader, val_sampler, criterion, config, device):
    model.eval()
    val_loss = 0.0
    top1 = 0.0
    top5 = 0.0

    with torch.no_grad():
        for sample in val_loader:
            x = sample['clip']
            x = x.to(device)
            t = sample['cls_id']
            t = t.to(device)

            h = model(x)

            val_loss += criterion(h, t).item()
            n, topk = accuracy(h, t, topk=(1, 5))
            top1 += topk[0].item()
            top5 += topk[1].item()

        # Horovod: use test_sampler to determine the number of examples in
        # this worker's partition.
        val_loss /= len(val_sampler)
        top1 /= len(val_sampler)
        top5 /= len(val_sampler)

        # Horovod: average metric values across workers.
        val_loss = metric_average(val_loss, 'avg_loss')
        top1 = metric_average(top1, 'avg_accuracy')
        top5 = metric_average(top5, 'avg_accuracy')

    return val_loss, top1, top5


def main():
    args = get_arguments()

    # Initialize Horovod
    hvd.init()
    torch.manual_seed(args.seed)

    # cuda
    if torch.cuda.is_available():
        device = 'cuda'
        # Pin GPU to be used to process local rank (one GPU per process)
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
    else:
        print('You have to use GPUs because training 3DCNN is computationally expensive.')
        sys.exit(1)

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

    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data, num_replicas=hvd.size(), rank=hvd.rank()
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_data, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        drop_last=True,
        sampler=train_sampler,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers,
        sampler=val_sampler,
        pin_memory=True
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

    # send models to device
    model.to(device)

    # set optimizer, lr_scheduler
    if CONFIG.optimizer == 'Adam':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG.learning_rate * hvd.size()
        )
    elif CONFIG.optimizer == 'SGD':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.SGD(
            model.parameters(),
            lr=CONFIG.learning_rate * hvd.size(),
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
            lr=CONFIG.learning_rate * hvd.size(),
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

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression
    )

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
        criterion = nn.CrossEntropyLoss(weight=get_class_weight().to(device))
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
        # training
        train_loss = train(
            model, train_loader, criterion, optimizer, CONFIG, device)
        train_losses.append(train_loss)

        # validation
        val_loss, top1, top5 = validation(
            model, val_loader, criterion, CONFIG, device)

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
            'epoch: {}\tloss train: {:.5f}\tloss val: {:.5f}\ttop1_accuracy: {:.5f}\ttop5_accuracy: {:.5f}'
            .format(epoch, train_losses[-1], val_losses[-1], top1_accuracy[-1], top5_accuracy[-1])
        )

    torch.save(
        model.module.state_dict(),
        os.path.join(CONFIG.result_path, 'final_model.prm')
    )


if __name__ == '__main__':
    main()
