import os
import time
import random
from typing import Generator
import numpy as np
import logging
import argparse
import shutil
from numpy.core.numeric import Inf
from numpy.testing._private.utils import requires_memory

import torch
from torch.autograd.variable import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.optim import optimizer
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.tensorboard import SummaryWriter
import glob
from util import config
from util.s3dis import Semantic3DTrain, Semantic3DTest
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn
from util import transform as t

import os
from model.pointtransformer.pointtransformer_seg import *

import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import sa_create
from util.data_util import data_prepare


test_mod = False


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/semantic3d/gan_config.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/gan_config.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.data_name == 'semantic3d':
        Semantic3DTrain(data_root=args.data_root, test_area=args.train_area)
        # Semantic3DTest(data_root=args.data_root, test_area=args.val_area)
    else:
        raise NotImplementedError()
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://localhost:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_loss
    args, best_loss = argss, Inf
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    generator = pointtransformer_gen_repro(c=args.fea_dim_g, k = args.classes)
    discriminator = pointtransformer_dis_repro(c=args.fea_dim_d, k = 1)
    if args.sync_bn:
       generator = nn.SyncBatchNorm.convert_sync_batchnorm(generator)
       discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
    criterion_pixelwise = torch.nn.L1Loss().cuda()
    criterion_GAN = torch.nn.BCELoss().cuda()

    #SGD
    optimizer_G = torch.optim.SGD(generator.parameters(), lr=args.g_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_D= torch.optim.SGD(discriminator.parameters(), lr=args.d_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler_G = lr_scheduler.MultiStepLR(optimizer_G, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.1)
    scheduler_D = lr_scheduler.MultiStepLR(optimizer_G, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.1)

    # #Adam
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr = args.adam_g_lr, betas=(args.adam_beta1, 0.999))
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = args.adam_d_lr, betas=(args.adam_beta1, 0.999))
    
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr = )
    if main_process():
        global logger, writer
        logger = get_logger()
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        generator = torch.nn.parallel.DistributedDataParallel(
            generator.cuda(),
            device_ids=[gpu],
            find_unused_parameters=True if "transformer" in args.arch else False
        )
        discriminator = torch.nn.parallel.DistributedDataParallel(
            discriminator.cuda(),
            device_ids=[gpu],
            find_unused_parameters=True if "transformer" in args.arch else False
        )

    else:
        generator = torch.nn.DataParallel(generator.cuda())
        discriminator = torch.nn.DataParallel(discriminator.cuda())
    
    valid = torch.ones((1,1)).cuda()
    fake = torch.zeros((1,1)).cuda()
    writer = SummaryWriter()

    train_transform = t.Compose([t.RandomScale([0.9, 1.1]), t.ChromaticAutoContrast(), t.ChromaticTranslation(), t.ChromaticJitter()]) #t.HueSaturationTranslation()])
    train_data = Semantic3DTrain( data_root=args.data_root, test_area=args.train_area, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop)
    if main_process():
            logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=collate_fn)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(train_loader, generator, discriminator, criterion_GAN, criterion_pixelwise, valid, fake, optimizer_D, optimizer_G, epoch, writer)
        scheduler_D.step()
        scheduler_G.step()
        epoch_log = epoch + 1

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/model/semantic3d_pth/gan_model_epoch_%d.pth'%(epoch_log)
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict_g': generator.state_dict(), 'state_dict_d': discriminator.state_dict(),
                         'optimizer_d': optimizer_D.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                        # 'scheduler_d': scheduler_G.state_dict(), 'scheduler_g':scheduler_G.state_dict()
                        },
                         filename)
            
def train(train_loader, generator, discriminator, criterion_GAN, criterion_pixelwise, valid, fake, optimizer_D, optimizer_G, epoch, writer = None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter_g = AverageMeter()
    loss_meter_d = AverageMeter()
    discriminator.train()
    generator.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (coord, feat, target, offset) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        data_time.update(time.time() - end)
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        target = torch.stack([target], dim=-1)
        output_g = generator(coord, target, offset)
        disc_fake_feat = torch.cat([target, output_g], dim = -1)
        disc_real_feat = torch.cat([target, feat], dim = -1)
        output_g_fake = discriminator(coord, disc_fake_feat, offset).squeeze()
        fake_l = Variable(torch.zeros_like(output_g_fake))
        real_l = Variable(torch.ones_like(output_g_fake))
        # print(real_l.shape)
        loss_point = criterion_pixelwise(output_g, feat)
        loss_g_fake = criterion_GAN(output_g_fake, real_l)
        print('loss_point', loss_point.item())
            # print('valid', valid)
        loss_g = args.lambda_L1*loss_point + args.lambda_G * loss_g_fake
            # loss_g = loss_point
        optimizer_G.zero_grad()
        loss_g.backward()
        optimizer_G.step()
        writer.add_scalar('Loss/loss_g', loss_g, epoch * i)
        optimizer_G.zero_grad()

        
        output_d_real = discriminator(coord, disc_real_feat.detach(), offset).squeeze()
        loss_d_real = criterion_GAN(output_d_real, real_l)
        # output_g_fake = discriminator([coord.detach(), target.detach(), offset.detach()])
        output_d_fake = discriminator(coord, disc_fake_feat.detach(), offset).squeeze()
        loss_d_fake = criterion_GAN(output_d_fake, fake_l)
        loss_d = (loss_d_real * 0.5 + loss_d_fake * 0.5) 
        # loss_d = loss_d_fake
        optimizer_D.zero_grad()
        loss_d.backward()
        optimizer_D.step()
        writer.add_scalar('Loss/loss_d', loss_d, epoch * i)

        print('Fake:', output_d_fake.mean().item(), '  loss_d_fake:', loss_d_fake.item())
        print('Real:', output_d_real.mean().item(), '  loss_d_real:', loss_d_real.item())
        n = coord.size(0)

        loss_meter_d.update(loss_d.item(), n)
        loss_meter_g.update(loss_g.item(), n)
     
        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] Loss_G {:.4f} Loss_D {:.4f} '.format(epoch+1, args.epochs, i + 1, len(train_loader), loss_g.item(), loss_d.item()))
        
    writer.add_scalar('mLoss/loss_g', loss_meter_g.avg, epoch)
    writer.add_scalar('mLoss/loss_d', loss_meter_d.avg, epoch)
    return loss_meter_d, loss_meter_g


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    # intersection_meter = AverageMeter()
    # union_meter = AverageMeter()
    # target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (coord, feat, target, offset) in enumerate(val_loader):
        data_time.update(time.time() - end)
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        # if target.shape[-1] == 1:
        #     target = target[:, 0]  # for cls
        target = torch.stack([target], dim=-1)
        with torch.no_grad():
            output = model([coord, target, offset])
        loss = criterion(output, feat)

        # output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter))


    if main_process():
        logger.info('Val result: mLoss {:.4f}.'.format(loss_meter.avg))
        for i in range(args.classes):
            logger.info('Class_{} Result: mLoss {:.4f}.'.format(i, loss_meter.avg))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return loss_meter.avg


if __name__ == '__main__':
    import gc
    gc.collect()
    main()

