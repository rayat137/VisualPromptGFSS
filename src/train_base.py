import os
import datetime
import random
import time
import cv2
import numpy as np
import logging
import argparse
import math
from visdom import Visdom
import os.path as osp
from shutil import copyfile

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from tensorboardX import SummaryWriter

from model.mask2former import get_model_m2former

from dataset import dataset #BaseData, transform
from dataset.dataset import get_val_loader
import dataset.transform as transform
from util import load_cfg_from_cfg_file, merge_cfg_from_list, merge_cfg_from_args
from util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, fast_intersection_and_union, setup_seed, get_logger, get_save_path, \
                                    is_same_model, fix_bn, sum_list, check_makedirs

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = '8'



def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='M2Former') # 
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/pascal/pascal_split0_vgg_base.yaml', help='config file') # coco/coco_split0_resnet50.yaml
    parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')    
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    cfg = merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_model_main(args):

    model = get_model_m2former(args, novel_fine_tune=False)
    # optimizer = torch.optim.AdamW([     
    #     {'params': model.parameters()}], lr=args.base_lr, weight_decay=args.weight_decay)
    #import pdb; pdb.set_trace()
    optimizer = torch.optim.AdamW([{'params':model.layer0.parameters(), 'lr': args.base_lr*0.1}, 
        {'params':model.layer1.parameters(), 'lr':args.base_lr*0.1},
        {'params':model.layer2.parameters(), 'lr':args.base_lr*0.1},
        {'params':model.layer3.parameters(), 'lr':args.base_lr*0.1},
        {'params':model.layer4.parameters(), 'lr':args.base_lr*0.1},
        {'params':model.bottleneck.parameters(), 'lr':args.base_lr*0.1},
        {'params':model.projection_output.parameters(), 'lr':args.base_lr*0.1},
        {'params':model.ppm.parameters(), 'lr':args.base_lr*0.1},
        {'params':model.pixel_decoder.parameters(), 'lr':args.base_lr},
        {'params':model.transformer.parameters(), 'lr':args.base_lr}
        ], lr=args.base_lr, weight_decay=args.weight_decay)

    if hasattr(model,'freeze_modules'):
        model.freeze_modules(model)

    if args.distributed:
        # Initialize Process Group
        dist.init_process_group(backend='nccl')
        print('args.local_rank: ', args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        model = model.cuda()
    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)
    if args.resume:
        resume_path = args.resume
        if os.path.isfile(resume_path):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try: 
                model.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            if main_process():       
                logger.info("=> no checkpoint found at '{}'".format(resume_path))

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    if main_process():
        print('Number of Parameters: %d' % (total_number))
        print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer

def main_process():
    return not args.distributed or (args.distributed and (args.local_rank == 0))

def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    args.distributed = True if torch.cuda.device_count() > 1 else False
    if main_process():
        print(args)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    
    if main_process():
        logger.info("=> creating model ...")
    model, optimizer = get_model_main(args)
    print(args.result_path)
    print(args.snapshot_path)
    if main_process():
        logger.info(model)
    if main_process() and args.viz:
        writer = SummaryWriter(args.result_path)

# ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    # Train
    train_transform = transform.Compose([
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    if args.data_set == 'pascal' or args.data_set == 'coco':
        train_data = dataset.BaseData(split=args.split, mode='train', data_root=args.data_root, data_list=args.train_list, \
                                    data_set=args.data_set, use_split_coco=args.use_split_coco, \
                                    transform=train_transform, main_process=main_process(), batch_size=args.batch_size, mask_classification= args.mask_loss)
    train_sampler = DistributedSampler(train_data) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, \
                                                pin_memory=True, sampler=train_sampler, drop_last=True, \
                                                shuffle=False if args.distributed else True)
    # Val
    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose([
                transform.Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=args.mean, std=args.std)])
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=args.mean, std=args.std)])
            
        
        if args.data_set == 'pascal' or args.data_set == 'coco':    
            val_data = dataset.BaseData(split=args.split, mode='val', data_root=args.data_root, data_list= args.val_list, \
                                        data_set=args.data_set, use_split_coco=args.use_split_coco, \
                                        transform=val_transform, main_process=main_process(), batch_size=args.batch_size_val)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=False, sampler=None)
        #val_loader = get_val_loader(args)
        if args.ori_resize:
            assert args.batch_size_val == 1

# ----------------------  TRAINVAL  ----------------------
    global best_miou, best_epoch, keep_epoch, val_num
    best_miou = 0.
    best_epoch = 0
    keep_epoch = 0
    val_num = 0

    start_time = time.time()
    max_iter = (args.epochs - args.start_epoch) * len(train_loader)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=max_iter,power=0.9)
    for epoch in range(args.start_epoch, args.epochs):
        if keep_epoch == args.stop_interval:
            break
        if args.fix_random_seed_val:
            setup_seed(args.manual_seed + epoch, args.seed_deterministic)

        epoch_log = epoch + 1
        keep_epoch += 1
        if args.distributed:
            train_sampler.set_epoch(epoch)    
        print("Best miou", best_miou, best_epoch)
        # ----------------------  TRAIN  ----------------------
        train(train_loader, model, optimizer, epoch, scheduler)

        # save model for <resuming>
        if (epoch % args.save_freq == 0) and (epoch > 0) and main_process():
            #filename = args.snapshot_path + '/epoch_{}.pth'.format(epoch)
            filename = args.snapshot_path + '/latest_model.pth'
            logger.info('Saving checkpoint to: ' + filename)
            if osp.exists(filename):
                os.remove(filename)            
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

        # -----------------------  VAL  -----------------------
        if args.evaluate and epoch%1==0:
            mIoU = validate(val_loader, model)            
            val_num += 1
            if main_process() and args.viz:
                writer.add_scalar('mIoU_val', mIoU, epoch_log)

        # save model for <testing>
            if (mIoU > best_miou):
                best_miou, best_epoch = mIoU, epoch
                keep_epoch = 0
                if main_process():
                    #filename = args.snapshot_path + '/train_epoch_' + str(epoch) + '_{:.4f}'.format(best_miou) + '.pth'
                    filename =  args.snapshot_path + '/best.pth'
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
                    #copyfile(filename, args.snapshot_path + '/best.pth')
            print("Best Miou and epoch: ",best_miou, best_epoch)
    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    if main_process():
        print('\nEpoch: {}/{} \t Total running time: {}'.format(epoch_log, args.epochs, total_time))
        print('The number of models validated: {}'.format(val_num))            
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Final Best Result   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print(args.arch + '\t Group:{} \t Best_mIoU:{:.4f} \t Best_step:{}'.format(args.split, best_miou, best_epoch))
        print('>'*80)
        print ('%s' % datetime.datetime.now())


def train(train_loader, model, optimizer, epoch, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    model.train()
    #model.freeze_bn()

    end = time.time()
    val_time = 0.
    max_iter = args.epochs * len(train_loader)
    is_mask_loss = args.mask_loss
    if main_process():
        print('Warmup: {}'.format(args.warmup))

    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end - val_time)
        current_iter = epoch * len(train_loader) + i + 1
        if is_mask_loss:
            classes_present = target[1]
            target = target[0]
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        loss = criterion(output,target) 


                
        #import pdb; pdb.set_trace()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        scheduler.step()

        n = input.size(0) # batch_size
        if not is_mask_loss:
            _,output = torch.max(output,dim=1)

            intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
            
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)  # allAcc
        
        loss_meter.update(loss.item(), n)

        batch_time.update(time.time() - end - val_time)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            if not is_mask_loss:
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '                      
                            'Loss {loss_meter.val:.4f} '
                            'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                            batch_time=batch_time,
                                                            data_time=data_time,
                                                            remain_time=remain_time,
                                                            loss_meter=loss_meter,
                                                            accuracy=accuracy))
                if args.viz:
                    writer.add_scalar('loss_train', loss_meter.val, current_iter)
            else:
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '                      
                            'Loss {loss_meter.val:.4f} '.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                            batch_time=batch_time,
                                                            data_time=data_time,
                                                            remain_time=remain_time,
                                                            loss_meter=loss_meter))

    if not is_mask_loss:
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        if main_process():
            logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU, mAcc, allAcc))


def validate(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    class_intersection_meter = [0]*(args.classes-1)
    class_union_meter = [0]*(args.classes-1)

    if args.manual_seed is not None and args.fix_random_seed_val:
        setup_seed(args.manual_seed, args.seed_deterministic)

    #criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    model.eval()
    end = time.time()
    val_start = end

    iter_num = 0
    is_mask_loss = args.mask_loss
    for i, logits in enumerate(val_loader):
        iter_num += 1
        data_time.update(time.time() - end)
        
        if args.batch_size_val == 1:
            input, target, ori_label = logits
            ori_label = ori_label.cuda(non_blocking=True)
        else:
            input, target = logits                
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        start_time = time.time()
        with torch.no_grad():
                
            output = model(input)

        model_time.update(time.time() - start_time)

        _,output = torch.max(output,dim=1)
        intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
        for idx in range(1,len(intersection)):
            class_intersection_meter[idx-1] += intersection[idx]
            class_union_meter[idx-1] += union[idx]

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        batch_time.update(time.time() - end)
        end = time.time()
        if ((iter_num % 100 == 0) or (iter_num == len(val_loader))) and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Accuracy {accuracy:.4f}.'.format(iter_num, len(val_loader),
                                                        data_time=data_time,
                                                        batch_time=batch_time,
                                                        accuracy=accuracy))
    val_time = time.time()-val_start
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i]/(class_union_meter[i]+ 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou*1.0 / len(class_intersection_meter)

    if main_process():
        logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou)) 
        for i in range(len(class_intersection_meter)):
            logger.info('Class_{} Result: iou_b {:.4f}.'.format(i+1, class_iou_class[i]))   
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
        print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, iter_num))

    return class_miou

if __name__ == '__main__':
    main()
