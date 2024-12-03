import argparse
import os
import time
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image

import numpy as np
import copy

#from classifier import Classifier
from classifier_m2former import Classifier
from dataset.dataset import get_val_loader
#from dataset.dataset_coco2pascal import get_val_loader
from util import get_model_dir, fast_intersection_and_union, setup_seed, resume_random_state, find_free_port, setup, \
    cleanup, get_cfg
import torch.nn.functional as F
from model.mask2former import get_model_m2former
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE, Isomap



def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    return get_cfg(parser)


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    print(f"==> Running evaluation script")

    setup_seed(args.manual_seed)

    # ========== Data  ==========
    val_loader, val_data = get_val_loader(args)
    base_class_names = []
    novel_class_names = []
    #import pdb; pdb.set_trace()
    # ========== Model  ==========
    model = get_model_m2former(args, novel_fine_tune=True).to(rank)
    print(model)
    model = torch.nn.DataParallel(model.cuda())

    root = args.ckpt_path
    print("=> Creating the model")
    #import pdb; pdb.set_trace()
    if args.ckpt_used is not None:
        filepath = os.path.join(root, f'{args.ckpt_used}.pth')
        assert os.path.isfile(filepath), filepath
        checkpoint = torch.load(filepath)
        state_dict = checkpoint['state_dict']
        state_dict = {(key if key.startswith('module.') else f"module.{key}"): value
        for key, value in state_dict.items()}
        #import pdb; pdb.set_trace()
        if args.one_vs_rest:
            state_dict['module.transformer.query_feat.weight'] = torch.cat((state_dict['module.transformer.query_feat.weight'], torch.randn(1,256).to(state_dict['module.transformer.query_feat.weight'].device) ))
            state_dict['module.transformer.query_embed.weight'] = torch.cat((state_dict['module.transformer.query_embed.weight'],torch.randn(1,256).to(state_dict['module.transformer.query_feat.weight'].device) ))
        else:
            state_dict['module.transformer.query_feat.weight'] = torch.cat((state_dict['module.transformer.query_feat.weight'], torch.randn(args.num_novel,256).to(state_dict['module.transformer.query_feat.weight'].device) ))
            state_dict['module.transformer.query_embed.weight'] = torch.cat((state_dict['module.transformer.query_embed.weight'],torch.randn(args.num_novel,256).to(state_dict['module.transformer.query_feat.weight'].device) ))
        model_dict = model.state_dict()
        for k in model_dict:
            if k in state_dict:
                model_dict[k] = state_dict[k]
        model.load_state_dict(model_dict)
        print("=> Loaded weight '{}'".format(filepath))
    else:
        print("=> Not loading anything")

    # ========== Test  ==========
    print("Base Classes:", base_class_names)
    print("Novel Classes:", novel_class_names)
    validate(args=args, val_loader=val_loader, model=model, model_dict=model_dict, base_class_names=base_class_names, novel_class_names=novel_class_names)

    #NOTE: UNCOMMENT WHEN DISTRIBUTED
    #cleanup()


def validate(args: argparse.Namespace, val_loader: torch.utils.data.DataLoader, model: DDP, model_dict, base_class_names, novel_class_names) -> Tuple[torch.tensor, torch.tensor]:
    print('\n==> Start testing ({} runs)'.format(args.n_runs), flush=True)
    random_state = setup_seed(args.manual_seed, return_old_state=True)
    #device = torch.device('cuda:{}'.format(dist.get_rank()))
    
    visualize_tsne = False
    compute_confusion_matrix = False
    

    device = torch.device('cuda:{}'.format(0))
    model.eval()
    c = 256 ##NOTE: MASK_DIM

    nb_episodes = len(val_loader) if args.test_num == -1 else int(args.test_num / args.batch_size_val)
    #nb_episodes = 1
    runtimes = torch.zeros(args.n_runs)
    base_mIoU, novel_mIoU = [torch.zeros(args.n_runs, device=device) for _ in range(2)]
    qualitative_results = False

    # ========== Perform the runs  ==========
    for run in range(args.n_runs):
        print('Run', run + 1, 'of', args.n_runs)
        # The order of classes in the following tensors is the same as the order of classifier (novels at last)
        cls_intersection = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_union = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_target = torch.zeros(args.num_classes_tr + args.num_classes_val)

        runtime = 0
        features_s, gt_s = None, None
        #import pdb; pdb.set_trace()
        print("Shots ", args.shot)
        if (not args.generate_new_support_set_for_each_task and args.multi_way):
            with torch.no_grad():
                #import pdb; pdb.set_trace()
                spprt_imgs, s_label, s_label_original = val_loader.dataset.generate_support([], remove_them_from_query_data_list=True)
                
                nb_episodes = len(val_loader) if args.test_num == -1 else nb_episodes  # Updates nb_episodes since some images were removed by generate_support
                ###NOTE: BASICALLY DIVIDE IT INTO MULTIPLE CHUNKS
                s_label = s_label.to(device, non_blocking=True)
                s_label_original = s_label_original.to(device, non_blocking=True)

                gt_s = s_label.view((args.num_classes_val, args.shot, args.image_size, args.image_size))
                base_gt_s = s_label_original.view((args.num_classes_val, args.shot, args.image_size, args.image_size))

                spprt_imgs = spprt_imgs.to(device, non_blocking=True)
                #import pdb; pdb.set_trace()
                features_s, multi_scale_s, last_layer = model.module.extract_features(spprt_imgs) #.detach().view((args.num_classes_val, args.shot, c, h, w))
                h = features_s.shape[-2]
                w = features_s.shape[-1]
                features_s = features_s.detach().view((args.num_classes_val, args.shot, c, h, w))

        if (args.multi_way and not args.use_transduction):  
 
            base_model = copy.deepcopy(model.module.transformer)
            #import pdb; pdb.set_trace()
            classifier = Classifier(args, n_tasks=args.batch_size_val, classifier=base_model, model_dict=model_dict)
            print("Optimizing M2Former")
            print(classifier.classifier.use_cross_attention)
            classifier.init_prototypes(features_s, gt_s)
            
            classifier.optimize_cross_entropy(features_s, multi_scale_s, gt_s, total_iters=args.adapt_iter)
       
        elif (args.multi_way and args.use_transduction): 
            # if run<1:
            #     continue
            
            base_model = copy.deepcopy(model.module.transformer)
            #import pdb; pdb.set_trace()
            classifier = Classifier(args, n_tasks=args.batch_size_val, classifier=base_model, model_dict=model_dict)
            print("Optimizing M2Former")

            classifier.init_prototypes(features_s, gt_s)
            #import pdb; pdb.set_trace()
            classifier.optimize_cross_entropy(features_s, multi_scale_s, gt_s, total_iters=args.pi_update_at[0])

            print("Begin Transduction")
            
        
        cls_intersection = torch.zeros(args.classes)
        cls_union = torch.zeros(args.classes)
        cls_target = torch.zeros(args.classes)

        episode = 1
        
        for ep in tqdm(range(nb_episodes), leave=True):
            t0 = time.time()        
            if args.multi_way:
                try:
                    loader_output = next(iter_loader)
                except (UnboundLocalError, StopIteration):
                    iter_loader = iter(val_loader)
                    loader_output = next(iter_loader)
                qry_img, q_label, q_valid_pix, img_path = loader_output
                
            with torch.no_grad():
                
                qry_img = qry_img.to(device, non_blocking=True)
                q_label = q_label.to(device, non_blocking=True)


                features_q, multi_scale_q, last_layer_q =  model.module.extract_features(qry_img)
                
                features_q = features_q.detach().unsqueeze(1)
                valid_pixels_q = q_valid_pix.unsqueeze(1).to(device)
                gt_q = q_label.unsqueeze(1)

            if args.use_transduction:
            
                new_base_model = copy.deepcopy(base_model)
                classifier = Classifier(args, n_tasks=args.batch_size_val, classifier=new_base_model, model_dict=None)
                print("Computing prior pi")
                classifier.compute_pi(features_q, multi_scale_q, valid_pixels_q, gt_q)  # gt_q won't be used in optimization if pi estimation strategy is self or uniform
                classifier.optimize_transduction(features_s, multi_scale_s, gt_s, features_q, multi_scale_q, valid_pixels_q,  gt_q=None)

            runtime += time.time() - t0

            # =========== Perform inference and compute metrics ===============

            with torch.no_grad():

                logits  = classifier.get_logits(multi_scale_q, features_q.squeeze())
                logits = logits.detach()
                if logits.dim() == 3:
                    logits = logits.unsqueeze(0)

                #import pdb; pdb.set_trace()
                probas = classifier.get_probas(logits)

            if probas.dim() == 4:
                probas = probas.unsqueeze(0)
            
            
            #import pdb; pdb.set_trace()      
            intersection, union, target = fast_intersection_and_union(probas, gt_q, novel_classes=args.classes-args.num_classes_tr)  # [batch_size_val, 1, num_classes]
            intersection, union, target = intersection.squeeze(1).cpu(), union.squeeze(1).cpu(), target.squeeze(1).cpu()


            if args.multi_way:
                #import pdb; pdb.set_trace()
                cls_intersection += intersection.sum(0)
                cls_union += union.sum(0)
                cls_target += target.sum(0)

            if args.debug_each_episode:
                base_count, novel_count, sum_base_IoU, sum_novel_IoU = 4 * [0]
                #for i, class_ in enumerate(all_class_list_pascal_to_coco):
                for i, class_ in enumerate(val_loader.dataset.all_classes):
                    if cls_union[i] == 0:
                        continue 
                    IoU = cls_intersection[i] / (cls_union[i] + 1e-10) 
                    print("Class {}: \t{:.4f}".format(class_, IoU))
                    #if class_ in pascal_to_coco:            
                    if class_ in val_loader.dataset.base_class_list:
                        sum_base_IoU += IoU
                        base_count += 1
                        #elif class_ in pascal_to_coco_novel:
                    elif class_ in val_loader.dataset.novel_class_list:
                        #import pdb; pdb.set_trace()
                        sum_novel_IoU += IoU
                        novel_count += 1
                #import pdb; pdb.set_trace()
                avg_base_IoU, avg_novel_IoU = sum_base_IoU / (base_count+1e-8), sum_novel_IoU / (novel_count+1e-8)
                print('Mean base IoU: {:.4f}, Mean novel IoU: {:.4f}'.format(avg_base_IoU, avg_novel_IoU), flush=True)
                print("Base:", base_mIoU, "Novel:", novel_mIoU)

        base_count, novel_count, sum_base_IoU, sum_novel_IoU = 4 * [0]
        print(val_loader.dataset.novel_class_list)
        #import pdb; pdb.set_trace()
        #for i, class_ in enumerate(all_class_list_pascal_to_coco):
        for i, class_ in enumerate(val_loader.dataset.all_classes):
            #print(class_)
            if cls_union[i] == 0:
                continue
            IoU = cls_intersection[i] / (cls_union[i] + 1e-10)
            print("Class {}: \t{:.4f}".format(class_, IoU))
            #if class_ in pascal_to_coco: 
            if class_ in val_loader.dataset.base_class_list:
                sum_base_IoU += IoU
                base_count += 1
            elif class_ in val_loader.dataset.novel_class_list:
            #lif class_ in pascal_to_coco_novel:
                
                sum_novel_IoU += IoU
                novel_count += 1
        
        print(sum_base_IoU, sum_novel_IoU)
        print(base_count, novel_count)
        avg_base_IoU, avg_novel_IoU = sum_base_IoU / (base_count+1e-8), sum_novel_IoU / (novel_count + 1e-8)
        print('Mean base IoU: {:.4f}, Mean novel IoU: {:.4f}'.format(avg_base_IoU, avg_novel_IoU), flush=True)
              
        base_mIoU[run], novel_mIoU[run] = avg_base_IoU, avg_novel_IoU



    agg_mIoU = (base_mIoU.mean() + novel_mIoU.mean()) / 2
    print('==>')
    print('Average of base mIoU: {:.4f}\tAverage of novel mIoU: {:.4f} \t(over {} runs)'.format(
        base_mIoU.mean(), novel_mIoU.mean(), args.n_runs))
    print("Base ", base_mIoU,"Novel", novel_mIoU )
    
    resume_random_state(random_state)
    return agg_mIoU


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    if args.debug:
        args.test_num = 64
        args.n_runs = 2

    world_size = len(args.gpus)
    distributed = world_size > 1
    assert not distributed, 'Testing should not be done in a distributed way'
    args.distributed = distributed
    args.port = find_free_port()
    print(world_size)
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    # mp.spawn(main_worker,
    #          args=(world_size, args),
    #          nprocs=world_size,
    #          join=True)args
    main_worker(rank=0, world_size=1, args=args )
