import argparse
import copy
import os
import random
from ast import literal_eval
from typing import Callable, Iterable, List, TypeVar
from typing import Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
import logging



A = TypeVar("A")
B = TypeVar("B")


def main_process(args: argparse.Namespace) -> bool:
    if args.distributed:
        rank = dist.get_rank()
        if rank == 0:
            return True
        else:
            return False
    else:
        return True


def setup(args: argparse.Namespace,
          rank: int,
          world_size: int) -> None:
    """
    Used for distributed learning
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    """
    Used for distributed learning
    """
    dist.destroy_process_group()


def find_free_port() -> int:
    """
    Used for distributed learning
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    """
    Used for multiprocessing
    """
    return list(map(fn, iter))


def get_next_run_id(args) -> int:
    return max(map(int, os.listdir(args.ckpt_path))) + 1


def get_model_dir(args: argparse.Namespace, run_id=None) -> str:
    """
    Obtain the directory to save/load the model
    """
    if run_id is None:
        run_id = args.load_model_id
    path = os.path.join(args.ckpt_path,
                        str(run_id),
                        args.data_name,
                        f'split{args.split}',
                        f'pspnet_resnet{args.layers}')
    return path


def save_model(name, savedir, epoch, model, optimizer):
    filename = os.path.join(savedir, '{}.pth'.format(name))
    print(f'Saving checkpoint to: {filename}')
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)


def to_one_hot(mask: torch.tensor,
               num_classes: int) -> torch.tensor:
    """
    inputs:
        mask : shape [b, shot, h, w]
        num_classes : Number of classes

    returns :
        one_hot_mask : shape [b, shot, num_class, h, w]
    """
    n_tasks, shot, h, w = mask.size()
    #NOTE: USE THIS FOR DIST
    #device = torch.device('cuda:{}'.format(dist.get_rank()))
    #device = torch.device('cuda:{}'.format(0))
    device = mask.device
    one_hot_mask = torch.zeros(n_tasks, shot, num_classes, h, w, device=device)
    new_mask = mask.unsqueeze(2).clone()
    new_mask[torch.where(new_mask == 255)] = 0  # Ignore_pixels are anyway filtered out in the losses
    one_hot_mask.scatter_(2, new_mask, 1).long()
    return one_hot_mask


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# I think some calls are redundant, in newer versions of PyTorch, torch.manual_seed should be sufficient.
def setup_seed(seed, return_old_state=False):
    old_state = list()
    if return_old_state:
        old_state.append(random.getstate())
        old_state.append(np.random.get_state())
        old_state.append(torch.get_rng_state())
        old_state.append(torch.cuda.get_rng_state())
        old_state.append(torch.cuda.get_rng_state_all())
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return old_state


def resume_random_state(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state(state[3])
    torch.cuda.set_rng_state_all(state[4])


def fast_intersection_and_union(probas: torch.Tensor,
                                target: torch.Tensor, novel_classes=20) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    n_task, shots, num_classes, h, w = probas.size()
    H, W = target.size()[-2:]
    #import pdb; pdb.set_trace()

    if (h, w) != (H, W):
        probas = F.interpolate(probas.view(n_task * shots, num_classes, h, w),
                               size=(H, W), mode='bilinear', align_corners=True).view(n_task, shots, num_classes, H, W)
        
    #import pdb; pdb.set_trace()

    # preds = torch.zeros(probas.shape[0],1, H, W).to(probas.device)
    # for batch in range(probas.shape[0]):
    #     #import pdb; pdb.set_trace()
    #     predicted_prob = probas[batch].squeeze()
    #     probas_temp = torch.argmax(predicted_prob,dim=0)
    #     indices = (probas_temp==0)
    #     predicted_prob[0,indices] = 0.1 * predicted_prob[0,indices]
    #     probas_back_novel_only =  torch.cat((predicted_prob[0].unsqueeze(0),predicted_prob[-novel_classes:]), dim=0)
    #     max_vals = torch.argmax(probas_back_novel_only, dim=0)
    #     max_vals[max_vals!=0] = max_vals[max_vals!=0] + (num_classes-novel_classes-1)
    #     probas_temp[indices] = max_vals[indices]
    #     preds[batch,0,:,:] = probas_temp
        
    

    preds = probas.argmax(2)  # [n_query, shot, H, W]

    # Pixels with target == 255 will be set to 0 in to_one_hot, we should ignore them
    valid_pixels = target.unsqueeze(2) != 255
    #import pdb; pdb.set_trace()
    target, preds = to_one_hot(target, num_classes), to_one_hot(preds.long(), num_classes)  # [n_task, shot, num_classes, H, W]

    area_intersection = (preds * target * valid_pixels).sum(dim=(3, 4))
    area_output = (preds * valid_pixels).sum(dim=(3, 4))
    area_target = (target * valid_pixels).sum(dim=(3, 4))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersection_and_union(preds: torch.tensor, target: torch.tensor, num_classes: int,
                           ignore_index=255) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    inputs:
        preds : shape [H, W]
        target : shape [H, W]
        num_classes : Number of classes

    returns :
        area_intersection : shape [num_class]
        area_union : shape [num_class]
        area_target : shape [num_class]
    """
    assert (preds.dim() in [1, 2, 3])
    assert preds.shape == target.shape
    preds = preds.view(-1)
    target = target.view(-1)
    preds[target == ignore_index] = ignore_index
    intersection = preds[preds == target]

    # This excludes ignore pixels (255) from the result, because of the max
    # Adding .float() because histc not working with long() on CPU
    area_intersection = torch.histc(intersection.float(), bins=num_classes, min=0, max=num_classes-1)
    area_output = torch.histc(preds.float(), bins=num_classes, min=0, max=num_classes-1)
    area_target = torch.histc(target.float(), bins=num_classes, min=0, max=num_classes-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def compute_wce(one_hot_gt, n_novel):
    n_novel_times_shot, n_classes = one_hot_gt.size()[1:3]
    shot = n_novel_times_shot // n_novel
    wce = torch.ones((1, n_novel_times_shot, n_classes, 1, 1), device=one_hot_gt.device)
    wce[:, :, 0, :, :] = 0.01 if shot == 1 else 0.15  # Increase relative coef of novel classes if labeled samples are scarce
    return wce


def get_cfg(parser):
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


# ======================================================================================================================
# ======== All following helper functions have been borrowed from from https://github.com/Jia-Research-Lab/PFENet ======
# ======================================================================================================================
class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def _decode_cfg_value(v):
    if not isinstance(v, str):
        return v
    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    casts = [(tuple, list), (list, tuple), (int, str)]
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg: CfgNode,
                        cfg_list: List[str]):
    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0, cfg_list
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, cfg[subkey], subkey, full_key
        )
        setattr(new_cfg, subkey, value)

    return new_cfg

def merge_cfg_from_args(cfg, args):
    args_dict = args.__dict__
    for k ,v in args_dict.items():
        if not k == 'config' or k == 'opts':
            cfg[k] = v
            
    return cfg


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9, index_split=-1, scale_lr=10., warmup=False, warmup_step=500):
    """poly learning rate policy"""
    if warmup and curr_iter < warmup_step:
        lr = base_lr * (0.1 + 0.9 * (curr_iter/warmup_step))
    else:
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

    # if curr_iter % 50 == 0:   
    #     print('Base LR: {:.4f}, Curr LR: {:.4f}, Warmup: {}.'.format(base_lr, lr, (warmup and curr_iter < warmup_step)))     

    for index, param_group in enumerate(optimizer.param_groups):
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr   # 10x LR




def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    #print(output.shape)
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def get_model_para_number(model):
    total_number = 0
    learnable_number = 0 
    for para in model.parameters():
        total_number += torch.numel(para)
        if para.requires_grad == True:
            learnable_number+= torch.numel(para)
    return total_number, learnable_number



def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger



def get_save_path(args):
    backbone_str = 'vgg' if args.vgg else 'resnet'+str(args.layers)
    args.snapshot_path = 'exp/{}/{}/split{}/{}/{}/mask_loss_{}/class_embed_{}/snapshot'.format(args.data_set, args.arch, args.split, backbone_str, args.batch_size, args.mask_loss, args.keep_class_embed)
    args.result_path = 'exp/{}/{}/split{}/{}/{}/mask_loss_{}/class_embed_{}/result'.format(args.data_set, args.arch, args.split, backbone_str, args.batch_size, args.mask_loss, args.keep_class_embed)

def is_same_model(model1, model2):
    flag = 0
    count = 0
    for k, v in model1.state_dict().items():
        model1_val = v
        model2_val = model2.state_dict()[k]
        if (model1_val==model2_val).all():
            pass
        else:
            flag+=1
            print('value of key <{}> mismatch'.format(k))
        count+=1

    return True if flag==0 else False


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def sum_list(list):
    sum = 0
    for item in list:
        sum += item
    return sum



def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
