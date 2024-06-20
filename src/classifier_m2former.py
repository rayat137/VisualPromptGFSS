from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

try:
    from .util import fast_intersection_and_union
except:
    from util import fast_intersection_and_union



#from src.util import compute_wce
# try:
#     from .util import compute_wce
# except:
#     from util import compute_wce
# try:
#     from .util import to_one_hot
# except:
#     from util import to_one_hot
import copy


def to_one_hot(mask: torch.tensor,
               num_classes: int, one_vs_rest=False) -> torch.tensor:
    """
    inputs:
        mask : shape [b, shot, h, w]
        num_classes : Number of classes

    returns :
        one_hot_mask : shape [b, shot, num_class, h, w]
    """
    n_tasks, shot, h, w = mask.size()

    device = torch.device('cuda:{}'.format(0))
    one_hot_mask = torch.zeros(n_tasks, shot, num_classes, h, w, device=device)
    new_mask = mask.unsqueeze(2).clone()
    new_mask[torch.where(new_mask == 255)] = 0  # Ignore_pixels are anyway filtered out in the losses
    if one_vs_rest:
        new_mask[torch.where(new_mask >= num_classes - 1)] = num_classes - 1 
    one_hot_mask.scatter_(2, new_mask, 1).long()
    return one_hot_mask


def compute_wce(one_hot_gt, n_novel):
    n_novel_times_shot, n_classes = one_hot_gt.size()[1:3]
    shot = n_novel_times_shot // n_novel
    wce = torch.ones((1, n_novel_times_shot, n_classes, 1, 1), device=one_hot_gt.device)
    wce[:, :, 0, :, :] = 0.01 if shot == 1 else 0.15   
    #wce[:, :, 0, :, :] = 0.01 if shot == 1 else 0.50 

    
    return wce

class Classifier(object):

    
    def __init__(self, args,n_tasks, classifier, model_dict):
        self.one_vs_rest = args.one_vs_rest
        self.num_base_classes_and_bg = args.num_classes_tr 
        self.model_dict = model_dict

        self.num_novel_classes = args.num_classes_val
        self.num_classes = self.num_base_classes_and_bg + self.num_novel_classes
        self.n_tasks = n_tasks
        self.classifier = classifier
        self.use_mlp = args.use_mlp
        self.multi_way = args.multi_way
        
        self.shot = args.shot
        if model_dict is None:
            self.classifier = classifier
        else:
            self.classifier.query_feat.weight = nn.Parameter(torch.cat((model_dict['module.transformer.query_feat.weight'][:self.num_base_classes_and_bg],torch.randn(args.num_novel,256).to(model_dict['module.transformer.query_feat.weight'].device) ), dim =0))
            self.classifier.query_embed.weight = nn.Parameter(torch.cat((model_dict['module.transformer.query_embed.weight'][:self.num_base_classes_and_bg] ,torch.randn(args.num_novel,256).to(model_dict['module.transformer.query_feat.weight'].device) ), dim =0))


        self.snapshot_weight = self.classifier.query_feat.weight[:args.num_classes_tr].squeeze(0).squeeze(0).clone()  # Snapshot of the model right after training, frozen
        self.snapshot_pe = self.classifier.query_embed.weight[:args.num_classes_tr].clone()
        self.snapshot_model =  copy.deepcopy(self.classifier)
        
    
        self.novel_weight, self.novel_pe = None, None
        self.pi, self.true_pi = None, None

        self.fine_tune_base_classifier = args.fine_tune_base_classifier
        self.lr = args.cls_lr
        self.adapt_iter = args.adapt_iter
        self.weights = args.weights
        self.pi_estimation_strategy = args.pi_estimation_strategy
        self.pi_update_at = args.pi_update_at
        

    @staticmethod
    def _valid_mean(t, valid_pixels, dim):
        s = (valid_pixels * t).sum(dim=dim)
        return s / (valid_pixels.sum(dim=dim) + 1e-10)

    def init_prototypes(self, features_s: torch.tensor, gt_s: torch.tensor) -> None:
        """
        inputs:
            features_s : shape [num_novel_classes, shot, c, h, w]
            gt_s : shape [num_novel_classes, shot, H, W]
        """
        # Downsample support masks
        print("Re-initialize the prototypes")

        ds_gt_s = F.interpolate(gt_s.float(), size=features_s.shape[-2:], mode='nearest')
        ds_gt_s = ds_gt_s.long().unsqueeze(2)  # [n_novel_classes, shot, 1, h, w]

        # Computing prototypes
        
        self.novel_weight = torch.zeros((features_s.size(2), self.num_novel_classes), device=features_s.device)
        for cls in range(self.num_base_classes_and_bg, self.num_classes):
            novel_mask = (ds_gt_s == cls) 
            novel_prototype = self._valid_mean(features_s, novel_mask, (0, 1, 3, 4))  # [c,]
            self.novel_weight[:, cls - self.num_base_classes_and_bg] = novel_prototype
   
        self.novel_weight /= self.novel_weight.norm(dim=0).unsqueeze(0) + 1e-10
        self.novel_weight = self.novel_weight.permute(1,0)

        query_feat = torch.cat((self.snapshot_weight ,self.novel_weight), dim =0)
        query_embed = torch.cat((self.snapshot_pe , torch.zeros_like(self.novel_weight, device=features_s.device )), dim =0)

        #Copy prototypes for each task
        self.classifier.query_feat.weight  = nn.Parameter(query_feat.unsqueeze(1).repeat(1, self.n_tasks,  1))
        self.classifier.query_embed.weight = nn.Parameter(query_embed.unsqueeze(1).repeat(1, self.n_tasks,  1))
        if self.use_mlp:
            self.classifier.novel_mask_embed = nn.Parameter(self.novel_weight.unsqueeze(0).repeat(self.n_tasks, 1,  1))

        ### REPEAT SAME FOR SNAPSHOT MODEL AS WELL
        self.snapshot_model.query_feat.weight  = nn.Parameter(query_feat.unsqueeze(1).repeat(1, self.n_tasks,  1))
        self.snapshot_model.query_embed.weight = nn.Parameter(query_embed.unsqueeze(1).repeat(1, self.n_tasks,  1))
        if self.use_mlp:
            self.snapshot_model.novel_mask_embed = nn.Parameter(self.novel_weight.unsqueeze(0).repeat(self.n_tasks, 1,  1))
        
       
    def get_attn(self, multi_scale_features, mask_features) -> torch.tensor:
        if mask_features.dim() == 3:
            mask_features= mask_features.unsqueeze(0)
        out = self.classifier(multi_scale_features, mask_features)
        multi_layer_attn = out['layerwise_attention']
        return multi_layer_attn

    def get_logits(self, multi_scale_features, mask_features, support=False) -> torch.tensor:
        """
        Computes logits for given features

        inputs:
            features : shape [1 or batch_size_val, num_novel_classes * shot or 1, c, h, w]

        returns :
            logits : shape [batch_size_val, num_novel_classes * shot or 1, num_classes, h, w]
        """
        if mask_features.dim() == 3:
            mask_features= mask_features.unsqueeze(0)
        if support:
            mask_features = mask_features.unsqueeze(0)
            for i in range(len(multi_scale_features)):
                if multi_scale_features[i].dim() <5:
                    multi_scale_features[i] = multi_scale_features[i].unsqueeze(0)
        else:
            mask_features = mask_features.unsqueeze(1)
            for i in range(len(multi_scale_features)): 
                if multi_scale_features[i].dim() <5:
                    multi_scale_features[i] = multi_scale_features[i].unsqueeze(1)

        out = self.classifier(multi_scale_features, mask_features)

        logits = out['pred_masks']
        
        return logits 

    @staticmethod
    def get_probas(logits: torch.tensor) -> torch.tensor:
        """
        inputs:
            logits : shape [batch_size_val, num_novel_classes * shot or 1, num_classes, h, w]

        returns :
            probas : shape [batch_size_val, num_novel_classes * shot or 1, num_classes, h, w]
        """
        #import pdb; pdb.set_trace()
        return torch.softmax(logits, dim=2)

    def get_base_snapshot_probas(self, features) -> torch.tensor:
        """
        Computes probability maps for given query features, using the snapshot of the base model right after the
        training. It only computes values for base classes.

        inputs:
            features : shape [batch_size_val, 1, c, h, w]

        returns :
            probas : shape [batch_size_val, 1, num_base_classes_and_bg, h, w]
        """
        logits = torch.einsum('bochw,cC->boChw', features, self.snapshot_weight) + self.snapshot_bias.view(1, 1, -1, 1, 1)
        return torch.softmax(logits, dim=2)

    def self_estimate_pi(self, logits_q: torch.tensor, unsqueezed_valid_pixels_q: torch.tensor) -> torch.tensor:
        """
        Estimates pi using model's prototypes

        inputs:
            logits_q : shape [batch_size_val, 1, c, h, w]
            unsqueezed_valid_pixels_q : shape [batch_size_val, 1, 1, h, w]

        returns :
            pi : shape [batch_size_val, num_classes]
        """
        probas = torch.softmax(logits_q, dim=2).detach()
        return self._valid_mean(probas, unsqueezed_valid_pixels_q, (1, 3, 4))

    def image_level_supervision_pi(self, logits_q: torch.tensor,
                                   unsqueezed_valid_pixels_q: torch.tensor) -> torch.tensor:
        """
        Estimates pi using model's prototypes and information about whether each class is present in a query image.

        inputs:
            features_q : shape [batch_size_val, 1, c, h, w]
            unsqueezed_valid_pixels_q : shape [batch_size_val, 1, 1, h, w]

        returns :
            pi : shape [batch_size_val, num_classes]
        """
        #logits_q = self.get_logits(features_q)
        absent_indices = torch.where(self.true_pi == 0)
        logits_q[absent_indices[0], :, absent_indices[1], :, :] = -torch.inf
        probas = torch.softmax(logits_q, dim=2).detach()
        return self._valid_mean(probas, unsqueezed_valid_pixels_q, (1, 3, 4))

    def compute_pi(self, features_q: torch.tensor, multi_scale_q,  valid_pixels_q: torch.tensor,
                   gt_q: torch.tensor = None) -> torch.tensor:
        """
        inputs:
            features_q : shape [batch_size_val, 1, c, h, w]
            valid_pixels_q : shape [batch_size_val, 1, h, w]
            gt_q : shape [batch_size_val, 1, H, W]
        """

        valid_pixels_q = F.interpolate(valid_pixels_q.float(), size=features_q.size()[-2:], mode='nearest').long()
        valid_pixels_q = valid_pixels_q.unsqueeze(2)
        logits_q = self.get_logits(multi_scale_q, features_q.squeeze(), support=False)

        if gt_q is not None:
            ds_gt_q = F.interpolate(gt_q.float(), size=features_q.size()[-2:], mode='nearest').long()
            one_hot_gt_q = to_one_hot(ds_gt_q, self.num_classes)  # [batch_size_val, shot, num_classes, h, w]
            self.true_pi = self._valid_mean(one_hot_gt_q, valid_pixels_q, (1, 3, 4))

        if self.pi_estimation_strategy == 'upperbound':
            self.pi = self.true_pi
        elif self.pi_estimation_strategy == 'self':
            self.pi = self.self_estimate_pi(logits_q, valid_pixels_q)
        elif self.pi_estimation_strategy == 'imglvl':
            self.pi = self.image_level_supervision_pi(logits_q, valid_pixels_q)
        elif self.pi_estimation_strategy == 'uniform':
            pi = 1 / self.num_classes
            self.pi = torch.full_like(self.true_pi, pi)  # [batch_size_val, num_classes]
        else:
            raise ValueError('pi_estimation_strategy is not implemented')

    def distillation_loss(self, curr_p: torch.tensor, snapshot_p: torch.tensor, valid_pixels: torch.tensor,
                          reduction: str = 'mean') -> torch.tensor:
        """
        inputs:
            curr_p : shape [batch_size_val, 1, num_classes, h, w]
            snapshot_p : shape [batch_size_val, 1, num_base_classes_and_bg, h, w]
            valid_pixels : shape [batch_size_val, 1, h, w]

        returns:
             kl : Distillation loss for the query
        """
        #import pdb; pdb.set_trace()
        adjusted_curr_p = curr_p.clone()[:, :, :self.num_base_classes_and_bg, ...]
        adjusted_curr_p[:, :, 0, ...] += curr_p[:, :, self.num_base_classes_and_bg:, ...].sum(dim=2)
        kl = (adjusted_curr_p * torch.log(1e-10 + adjusted_curr_p / (1e-10 + snapshot_p))).sum(dim=2)
        #import pdb; pdb.set_trace()
        kl = self._valid_mean(kl, valid_pixels, (1, 2, 3))
        if reduction == 'sum':
            kl = kl.sum(0)
        elif reduction == 'mean':
            kl = kl.mean(0)
        return kl
    


    def get_entropies(self, valid_pixels: torch.tensor, probas: torch.tensor,
                      reduction: str = 'mean') -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        inputs:
            valid_pixels: shape [batch_size_val, 1, h, w]
            probas : shape [batch_size_val, 1, num_classes, h, w]

        returns:
            d_kl : Classes proportion kl
            entropy : Entropy of predictions
            marginal : Current marginal distribution over labels [batch_size_val, num_classes]
        """
        #import pdb; pdb.set_trace()
        entropy = - (probas * torch.log(probas + 1e-10)).sum(2)
        entropy = self._valid_mean(entropy, valid_pixels, (1, 2, 3))
        marginal = self._valid_mean(probas, valid_pixels.unsqueeze(2), (1, 3, 4))

        d_kl = (marginal * torch.log(1e-10 + marginal / (self.pi + 1e-10))).sum(1)

        if reduction == 'sum':
            entropy = entropy.sum(0)
            d_kl = d_kl.sum(0)
            assert not torch.isnan(entropy), entropy
            assert not torch.isnan(d_kl), d_kl
        elif reduction == 'mean':
            entropy = entropy.mean(0)
            d_kl = d_kl.mean(0)
        return d_kl, entropy, marginal

    

    def get_ce(self, probas: torch.tensor, valid_pixels: torch.tensor, one_hot_gt: torch.tensor,
               reduction: str = 'mean') -> torch.tensor:
        """
        inputs:
            probas : shape [batch_size_val, num_novel_classes * shot, c, h, w]
            valid_pixels : shape [1, num_novel_classes * shot, h, w]
            one_hot_gt: shape [1, num_novel_classes * shot, num_classes, h, w]

        returns:
             ce : Cross-Entropy between one_hot_gt and probas
        """
        #import pdb; pdb.set_trace()

        probas = probas.clone()
        probas[:, :, 0, ...] += probas[:, :, 1:self.num_base_classes_and_bg, ...].sum(dim=2)
        probas[:, :, 1:self.num_base_classes_and_bg, ...] = 0.

        ce = - (one_hot_gt * torch.log(probas + 1e-10))
        #import pdb; pdb.set_trace()
        ce = (ce * compute_wce(one_hot_gt, self.num_novel_classes)).sum(2)
        ce = self._valid_mean(ce, valid_pixels, (1, 2, 3))  # [batch_size_val,]

        if reduction == 'sum':
            ce = ce.sum(0)
        elif reduction == 'mean':
            ce = ce.mean(0)
        return ce
    

    def optimize_cross_entropy(self, features_s, multi_scale_s, gt_s: torch.tensor, total_iters=100) -> torch.tensor:
        l1, l2, l3, l4 = self.weights

        split_shots = False
        
        ##NOTE: MODIFY THIS FOR TIME BEING
        params = [{'params': self.classifier.query_feat.weight}]
        params.append({'params': self.classifier.query_embed.weight})
        params.append({'params': self.classifier.novel_mask_embed })
        if self.classifier.use_cross_attention:
            params.append({'params': self.classifier.base_to_novel_cross_attention_layers.parameters()})

        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.05)
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        features_s = features_s.flatten(0, 1).unsqueeze(0)
        gt_s = gt_s.flatten(0, 1).unsqueeze(0)
        ds_gt_s = F.interpolate(gt_s.float(), size=features_s.size()[-2:], mode='nearest').long()

        one_hot_gt_s = to_one_hot(ds_gt_s, self.num_classes, self.one_vs_rest)  # [1, num_novel_classes * shot, num_classes, h, w]
        valid_pixels_s = (ds_gt_s != 255).float()  

        if split_shots:
            mid_iters = 2
            stride = features_s.shape[1]//mid_iters #self.shot
            #mid_iters = 2 #self.shot
            
            multi_scale_s_new = []
            for i in range(mid_iters):
                temp_multi_scale_s = []
                for j in range(len(multi_scale_s)):
                    temp_multi_scale_s.append(multi_scale_s[j][i*stride:i*stride + stride])

                multi_scale_s_new.append(temp_multi_scale_s)


        max_range =  features_s.shape[1]
        print("Total iterations for ce-optimization", total_iters)
        for iteration in range(total_iters):
            if split_shots:
               for i in range(mid_iters):
                    features_s_new = features_s[:,i*stride:i*stride + stride,:, :, :]
                    one_hot_gt_s_new = one_hot_gt_s[:,i*stride:i*stride + stride,:,:,:]
                    valid_pixels_s_new = valid_pixels_s[:,i*stride:i*stride + stride,:,:]
                    # Using 16-bit floating point
                    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                        logits_s = self.get_logits(multi_scale_s_new[i], features_s_new.squeeze(), support=True)                        
                        proba_s = self.get_probas(logits_s)
                        loss = self.get_ce(proba_s, valid_pixels_s_new, one_hot_gt_s_new, reduction='none') 
                        optimizer.zero_grad()
                        scaler.scale(loss.sum(0)).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    
            else:
                # Using 16-bit floating point
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                    logits_s = self.get_logits(multi_scale_s, features_s.squeeze(), support=True)#.unsqueeze(0)
                    if not self.multi_way:
                        logits_s = logits_s.unsqueeze(0)
                    proba_s = self.get_probas(logits_s)
                    loss =  self.get_ce(proba_s, valid_pixels_s, one_hot_gt_s, reduction='none') #+ 0.01* torch.mean(torch.mean(self.classifier.novel_mask_embed,dim=-1),dim=-1)
                    optimizer.zero_grad()
                    scaler.scale(loss.sum(0)).backward()
                    scaler.step(optimizer)
                    scaler.update()

    
    
    def optimize_transduction(self, features_s, multi_scale_s, gt_s: torch.tensor, features_q, multi_scale_q, valid_pixels_q,   gt_q=None) -> torch.tensor:
        if gt_q is not None:

            writer = SummaryWriter()
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        split_shots = False
        l1, l2, l3, l4 = self.weights
        print(l1, l2, l3, l4)
        #import pdb; pdb.set_trace()
        params = []
        params = [{'params': self.classifier.query_feat.weight}]
        params.append({'params': self.classifier.novel_mask_embed })
        params.append({'params': self.classifier.query_embed.weight})
        #import pdb; pdb.set_trace()
        if self.classifier.use_cross_attention:
            params.append({'params': self.classifier.base_to_novel_cross_attention_layers.parameters()})
        
        # Flatten the dimensions of different novel classes and shots
        features_s = features_s.flatten(0, 1).unsqueeze(0)
        gt_s = gt_s.flatten(0, 1).unsqueeze(0)

        ds_gt_s = F.interpolate(gt_s.float(), size=features_s.size()[-2:], mode='nearest').long()

        one_hot_gt_s = to_one_hot(ds_gt_s, self.num_classes, self.one_vs_rest)  # [1, num_novel_classes * shot, num_classes, h, w]
        valid_pixels_s = (ds_gt_s != 255).float() 

        
        valid_pixels_q = F.interpolate(valid_pixels_q.float(), size=features_q.size()[-2:], mode='nearest').long()

        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.05)
        print(self.pi_update_at)

        if split_shots:
            mid_iters = 2
            stride = features_s.shape[1]//mid_iters#self.shot
            
            multi_scale_s_new = []
            for i in range(mid_iters):
                temp_multi_scale_s = []
                for j in range(len(multi_scale_s)):
                    temp_multi_scale_s.append(multi_scale_s[j][i*stride:i*stride + stride])

                multi_scale_s_new.append(temp_multi_scale_s)

        for iteration in range(self.pi_update_at[0], self.adapt_iter):
            if split_shots:
                for i in range(mid_iters):
                    features_s_new = features_s[:,i*stride:i*stride + stride,:, :, :]
                    one_hot_gt_s_new = one_hot_gt_s[:,i*stride:i*stride + stride,:,:,:]
                    valid_pixels_s_new = valid_pixels_s[:,i*stride:i*stride + stride,:,:]
                    # Using 16-bit floating point
                    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                        logits_s = self.get_logits(multi_scale_s_new[i], features_s_new.squeeze(), support=True)      
                        proba_s = self.get_probas(logits_s)
                        ce = self.get_ce(proba_s, valid_pixels_s_new, one_hot_gt_s_new, reduction='none') 
                        logits_q = self.get_logits(multi_scale_q, features_q.squeeze(), support=False)

                        proba_q = self.get_probas(logits_q)
                        with torch.no_grad():
                            out = self.snapshot_model(multi_scale_q, features_q)
                            snapshot_logit_q = out['pred_masks']
                            snapshot_proba_q = self.get_probas(snapshot_logit_q)
                            snapshot_proba_q = snapshot_proba_q[:,:,:self.num_base_classes_and_bg,:,:]

                        distillation = self.distillation_loss(proba_q, snapshot_proba_q , valid_pixels_q, reduction='none')

                        d_kl, entropy, marginal = self.get_entropies(valid_pixels_q, proba_q, reduction='none')
                        loss =  l1 * ce + l2 * d_kl + l3 * entropy + l4 * distillation

                        optimizer.zero_grad()

                        scaler.scale(loss.sum(0)).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                        del(logits_s)
                        del(proba_s)
                        if iteration >= self.pi_update_at[0]:
                            del(logits_q)
                            del(proba_q)
                        torch.cuda.empty_cache()
            else:
                # Using 16-bit floating point
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                    logits_s = self.get_logits(multi_scale_s, features_s.squeeze(),  support=True)
                    proba_s = self.get_probas(logits_s)
                    ce = self.get_ce(proba_s, valid_pixels_s, one_hot_gt_s, reduction='none') 

                    logits_q = self.get_logits(multi_scale_q, features_q.squeeze(),   support=False)
                    proba_q = self.get_probas(logits_q)
                    with torch.no_grad():
                        out = self.snapshot_model(multi_scale_q, features_q)
                        snapshot_logit_q = out['pred_masks']
                        snapshot_proba_q = self.get_probas(snapshot_logit_q)
                        snapshot_proba_q = snapshot_proba_q[:,:,:self.num_base_classes_and_bg,:,:]
                    distillation = self.distillation_loss(proba_q, snapshot_proba_q , valid_pixels_q, reduction='none')
                    d_kl, entropy, marginal = self.get_entropies(valid_pixels_q, proba_q, reduction='none')
                    #print("Apply all loss")
                    loss =  l1 * ce + l2 * d_kl + l3 * entropy + l4 * distillation 
                    optimizer.zero_grad()

                    #For mixed precision
                    scaler.scale(loss.sum(0)).backward()
                    scaler.step(optimizer)
                    scaler.update()

            # Update pi, if there is a second update in pi_update_at list 
            if (iteration + 1) in self.pi_update_at and (self.pi_estimation_strategy == 'self') and (l2 != 0):
                #print("Updating pi")
                self.compute_pi(features_q, multi_scale_q, valid_pixels_q)

    
    
    
    