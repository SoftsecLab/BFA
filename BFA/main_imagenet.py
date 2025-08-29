"""
这段代码的主要功能是：在imagenet数据集上训练指定的模型，同时在验证集合上评估模型性能；
支持分布式训练：可以在多个GPU上进行训练
学习了调度：支持手动调整学习率和余弦学习率
模型保存与加载：支持仅评估模型而不进行训练
注意力机制：支持在模型中添加不同的注意力模块
"""
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from torch.utils.data import Subset

from util import AverageMeter, ProgressMeter, accuracy, parse_gpus
from checkpoint import save_checkpoint, load_checkpoint
from thop import profile
from networks.imagenet import create_net

# argparse:用于解析命令行参数
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')# 数据路径
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')# 模型架构加载
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')# 数据加载线程数
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')# 训练总轮数
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')# 表示从哪一个轮次开始训练
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')# 批量大小
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')# 初始学习率
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')# 优化器动量
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')# 权重衰减 L2正则化
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')# 打印训练信息的频率
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')# 从检查点恢复训练路径
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')# 仅仅评估模型不训练
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')# 使用预训练模型
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')# 表示随机种子
parser.add_argument('--gpu', default="0",
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument("--ckpt", default="./ckpts/", 
                    help="folder to output checkpoints")
parser.add_argument("--attention_type", type=str, default="none",
                    help="attention type (possible choices none | se | cbam | simam)")# 选择注意力类型
parser.add_argument("--attention_param", type=float, default=4,
                    help="attention parameter (reduction factor in se and cbam, e_lambda in simam)")
parser.add_argument("--log_freq", type=int, default=500,
                    help="log frequency to file")
parser.add_argument("--cos_lr", action='store_true',
                    help='use cosine learning rate')# 使用余弦学习率调度器
parser.add_argument("--save_weights", default=None, type=str, metavar='PATH',
                    help='save weights by CPU for mmdetection') # 权重保存路径


best_acc1 = 0

#  解析命令行参数并初始化训练环境，设置随机种子确保实验的可复现性
#  创建检查点文件（用于保存模型权重和日志）
#  根据是否启用分布式训练，调用main_worker函数
def main():
    args = parser.parse_args()# 解析命令行参数并存储在 args 中

    # 根据模型架构和注意力机制类型，生成检查点文件夹的名称
    args.ckpt += "imagenet"
    args.ckpt += "-" + args.arch
    if args.attention_type.lower() != "none":
        args.ckpt += "-" + args.attention_type
    if args.attention_type.lower() != "none":
        args.ckpt += "-param" + str(args.attention_param)

    # 解析 GPU ID 并设置设备（GPU 或 CPU）
    args.gpu = parse_gpus(args.gpu)
    if args.gpu is not None:
        args.device = torch.device("cuda:{}".format(args.gpu[0]))
    else:
        args.device = torch.device("cpu")

    # 设置随机种子以确保实验可复现，并启用 CUDNN 确定性模式
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        args.ckpt += '-seed' + str(args.seed)

    # 如果检查点文件夹不存在，则创建它。
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    # 如果指定了 GPU，则禁用数据并行。
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    # 如果分布式训练的 URL 是 env://，则从环境变量中获取 WORLD_SIZE。
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # 判断是否启用分布式训练。
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # 如果启用多进程分布式训练，则启动多个进程；否则直接调用 main_worker 函数。
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

# 主工作函数
# 初始化分布式训练（如果启用）；创建模型并计算模型的FLOPs和参数量
# 加载数据集并数据预处理；定义损失函数（交叉熵）和优化器（SGD）
# 如果制定了恢复训练参数，加载之前保存的模型和优化器状态；调用训练和验证函数进行模型训练和评估；保存模型最佳检查点
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1 # 定义全局变量 best_acc1，用于保存最佳验证准确率。

    if args.gpu is not None:# 打印使用的 GPU ID。
        print("Use GPU: {} for training".format(args.gpu))

    # 初始化分布式训练
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model 创建模型，调用的是_init_中的函数
    model = create_net(args)

    # 计算模型的 FLOPs 和参数量。
    x = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(x,))

    # 打印模型的参数量和 FLOPs。
    print("model [%s] - params: %.6fM" % (args.arch, params / 1e6))
    print("model [%s] - FLOPs: %.6fG" % (args.arch, flops / 1e9))

    # 打开日志文件并记录模型信息。
    log_file = os.path.join(args.ckpt, "log.txt")

    if os.path.exists(log_file):
        args.log_file = open(log_file, mode="a")
    else:
        args.log_file = open(log_file, mode="w")
        args.log_file.write("Network - " + args.arch + "\n")
        args.log_file.write("Attention Module - " + args.attention_type + "\n")
        # args.log_file.write("Params - " % str(params) + "\n")
        args.log_file.write(f"Params - {params}\n")
        # args.log_file.write("FLOPs - " % str(flops) + "\n")
        args.log_file.write(f"FLOPs - {flops}\n")
        args.log_file.write("--------------------------------------------------" + "\n")

    args.log_file.close()


    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.device)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.device)
        model = model.to(args.gpu[0])
        model = torch.nn.DataParallel(model, args.gpu) 

    print(model)

    # define loss function (criterion) and optimizer
    # 定义交叉熵损失函数，并将其移动到 GPU。
    # 定义 SGD 优化器，包含学习率、动量和权重衰减等参数
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # 如果指定了恢复训练的参数，载入之前保存的模型和优化器状态。
    if args.resume:
        model, optimizer, best_acc1, start_epoch = load_checkpoint(args, model, optimizer)
        args.start_epoch = start_epoch

    cudnn.benchmark = True

    # 指定训练和验证数据的目录，并定义数据归一化的参数。
    # Data loading code
    traindir = os.path.join(args.data, 'train')# 将训练数据的路径设置为 args.data/train。args.data 是命令行参数中指定的数据集根目录。
    valdir = os.path.join(args.data, 'val')# 将验证数据的路径设置为 args.data/val
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) # 定义数据归一化的参数。mean 和 std 是 ImageNet 数据集的均值和标准差，用于将图像数据归一化到 [-1, 1] 范围。
    # 使用 ImageFolder 加载训练数据，并进行一系列数据增强和预处理。
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # 使用双三次插值进行上采样
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.Resize(256),  # 首先上采样（默认使用双线插值）调整图像大小到 256x256，让tiny-imagenet适应imagenet
            transforms.RandomResizedCrop(224),# 随机裁剪图像并调整大小为 224x224。
            transforms.RandomHorizontalFlip(),# 随机水平翻转图像（数据增强）。
            transforms.ToTensor(),# 将图像转换为 PyTorch 张量。
            normalize,# 对图像进行归一化。
        ]))
    # 如果启用了分布式训练，则使用 DistributedSampler 对训练数据进行采样，以确保每个进程获取不同的数据子集。
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    # 创建训练数据加载器。
    # shuffle：是否打乱数据。如果启用了分布式训练（train_sampler 不为 None），则不进行打乱。
    # 批量大小，由命令行参数 args.batch_size 指定。
    # num_workers：数据加载的线程数
    # pin_memory：将数据加载到固定内存中，以加速 GPU 数据传输。
    # sampler：数据采样器，用于分布式训练。

    # 只使用前 500 个样本
    indices = list(range(500))  # 前 500 个样本的索引
    # train_dataset= Subset(train_dataset, indices)# 如果想取消数量限制，就注释掉这一行


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    # 类似地，创建验证数据加载器，只进行必要的预处理。
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([# 对验证数据进行预处理：
            # 使用双三次插值进行上采样
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.Resize(256),# 首先上采样（默认使用双线插值）调整图像大小到 256x256，让tiny-imagenet适应imagenet
            transforms.Resize(256),# 将图像调整为 256x256。
            transforms.CenterCrop(224),# 从中心裁剪出 224x224 的图像。
            transforms.ToTensor(),# 将图像转换为 PyTorch 张量。
            normalize,# 对图像进行归一化。
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)# shuffle=False：验证数据不需要打乱。其他与训练集相同
    # 如果指定了保存权重的路径，保存"去并行化"的模型权重。
    if args.save_weights is not None: # "deparallelize" saved weights，如果指定了 args.save_weights，则保存模型的权重。
        print("=> saving 'deparallelized' weights [%s]" % args.save_weights)
        model = model.module# 如果模型是并行化的（如 DataParallel 或 DistributedDataParallel），则提取原始模型。
        model = model.cpu()# 将模型权重移动到 CPU。
        torch.save({'state_dict': model.state_dict()}, args.save_weights, _use_new_zipfile_serialization=False)# torch.save：保存模型的权重到指定路径。
        return

    #  评估和学习率调度
    # 如果指定了评估模式，调用验证函数并退出。
    if args.evaluate:
        args.log_file = open(log_file, mode="a")
        validate(val_loader, model, criterion, args)
        args.log_file.close()
        return
    # 如果使用余弦学习率调度，在每个 epoch 开始时更新学习率。
    if args.cos_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        for epoch in range(args.start_epoch):
            scheduler.step()

    #  主训练循环
    for epoch in range(args.start_epoch, args.epochs):
        
        args.log_file = open(log_file, mode="a")

        if args.distributed:
            train_sampler.set_epoch(epoch)

        if(not args.cos_lr):# 调用的是最下面的函数
            adjust_learning_rate(optimizer, epoch, args)
        else:
            scheduler.step()
            print('[%03d] %.5f'%(epoch, scheduler.get_lr()[0]))


        # train for one epoch，调用的是下面的函数
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        args.log_file.close()

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):

            save_checkpoint({
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_acc": best_acc1,
                "optimizer" : optimizer.state_dict(),
                }, is_best, epoch, save_path=args.ckpt)

# 这个函数负责执行一个训练周期，处理数据，计算损失，更新模型权重，并记录训练过程中的各种指标。
# 接收训练数据加载器、模型、损失函数、优化器、当前 epoch 和其他参数
def train(train_loader, model, criterion, optimizer, epoch, args):
    # 使用 AverageMeter 类来跟踪训练过程中的时间、损失和准确率等指标。
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter( # 创建一个进度条，用于显示当前 epoch 的训练状态。
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    param_groups = optimizer.param_groups[0]
    curr_lr = param_groups["lr"]

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):# 遍历训练数据加载器，获取每个批次的图像和目标标签。
        # measure data loading time记录数据加载时间。
        data_time.update(time.time() - end)

        # 将数据迁移到指定的 GPU 上。
        if args.gpu is not None:
            images = images.to(args.device, non_blocking=True)
        if torch.cuda.is_available():
            target = target.to(args.device, non_blocking=True)

        # compute output计算模型输出和损失值。
        output = model(images)
        loss = criterion(output, target)

        # accuracy是工具函数
        # measure accuracy and record loss计算 top-1 和 top-5 准确率，并更新指标
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step清零梯度，计算梯度并更新模型权重。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time更新批次处理时间。
        batch_time.update(time.time() - end)
        end = time.time()

        # 定期打印训练进度和当前学习率。
        if i % args.print_freq == 0:
            epoch_msg = progress.get_message(i)
            epoch_msg += ("\tLr  {:.4f}".format(curr_lr))
            print(epoch_msg)

        if i % args.log_freq == 0:
            args.log_file.write(epoch_msg + "\n")
 # 这个函数负责在验证集上评估模型性能，不会更新模型权重。与 train 函数类似，初始化用于跟踪时间、损失和准确率的指标。
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode设置模型为评估模式，禁用 dropout 和 batch normalization。
    model.eval()
    # 在不计算梯度的情况下遍历验证数据。
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
           
            if args.gpu is not None:
                images = images.to(args.device, non_blocking=True)
            if torch.cuda.is_available():
                target = target.to(args.device, non_blocking=True)

            # compute outputs 计算模型输出和损失，与训练过程相同
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss记录损失和准确率，并更新相应的指标。
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                epoch_msg = progress.get_message(i)
                print(epoch_msg)

        # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))

        epoch_msg = '----------- Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} -----------'.format(top1=top1, top5=top5)

        print(epoch_msg)

        args.log_file.write(epoch_msg + "\n")


    return top1.avg

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # 每经过 30 个 epoch，学习率减小为原来的 10%。
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()