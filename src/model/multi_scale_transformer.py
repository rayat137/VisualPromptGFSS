from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.functional import normalize, interpolate, sigmoid
import math
from collections import OrderedDict
from torch.nn import functional as F
import copy


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask = None,
                     tgt_key_padding_mask = None,
                     query_pos = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask = None,
                    tgt_key_padding_mask = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask = None,
                tgt_key_padding_mask = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)



class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, novel=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        if not novel:
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask,
                     memory_key_padding_mask,
                     pos,
                     query_pos,
                     novel=False):
        
        #import pdb; pdb.set_trace()
        tgt2, attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)#[0]
        if novel:
            tgt = tgt + tgt2
        else:    
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)
        
        return tgt, attn_weights

    def forward_pre(self, tgt, memory,
                    memory_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None,
                    novel = False):
        if not novel:
            tgt2 = self.norm(tgt)
        tgt2, attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)#[0]
        if novel:
            tgt = tgt + tgt2
        else:    
            tgt = tgt + self.dropout(tgt2)

        return tgt, attn_weights

    def forward(self, tgt, memory,
                memory_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None,
                novel = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos, novel=novel)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos, novel=novel)



class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool = False,
        novel_finetune=False, 
        num_novel = 0,
        use_mlp = True,
        num_base_classes =0, 
        shots = 5
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """

        super().__init__()

        self.mask_classification = mask_classification
        self.novel_finetune = novel_finetune
        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.num_classes = num_classes
        self.num_novel_classes = num_novel
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.use_mlp = use_mlp
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.use_cross_attention = True 
        self.num_base_classes = num_base_classes
        self.shots = shots
        if self.novel_finetune:
            if self.use_mlp:
                self.novel_mask_embed = nn.Parameter(torch.randn(num_novel,mask_dim, requires_grad=True))

            if self.use_cross_attention:  
                self.base_to_novel_cross_attention_layers = CrossAttentionLayer(
                            d_model=hidden_dim,
                            nhead=nheads,
                            dropout=0.0,
                            normalize_before=pre_norm,
                            novel = False
                        )   
        
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features

        # FREEZE THE QUERY FEATURES FOR CLIP EMBEDDING
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        #self.query_feat.weight.requires_grad=False
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                nn.init.xavier_uniform_(self.input_proj[-1].weight)
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.use_mlp:
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)


    def forward(self, x, mask_features,  mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        attn=None
        # disable mask, it does not affect performance
        del mask

        #import pdb; pdb.set_trace()
        
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])

            if self.novel_finetune:      
                b,o,c,h,w = mask_features.shape
                if (b==1 and o==1) or (b==1 and o!=1):
                    pos.append(self.pe_layer(x[i].squeeze(0), None).flatten(-2))
                    src.append(self.input_proj[i](x[i]).flatten(-2).squeeze(0) + self.level_embed.weight[i][None, :, None])
                else:
                    pos.append(self.pe_layer(x[i].squeeze(1), None).flatten(-2))
                    src.append(self.input_proj[i](x[i]).flatten(-2).squeeze(1) + self.level_embed.weight[i][None, :, None])
            else:
                pos.append(self.pe_layer(x[i], None).flatten(-2))
                src.append(self.input_proj[i](x[i]).flatten(-2) + self.level_embed.weight[i][None, :, None])
            
            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        if self.novel_finetune:
            #import pdb; pdb.set_trace()
            query_embed = self.query_embed.weight
            output = self.query_feat.weight
            if b==1 and o!=1:
                query_embed = query_embed.repeat(1, o, 1)
                output = output.repeat(1, o, 1)
        else:
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            

        predictions_mask = []
        # prediction heads on learnable query features
        if (self.novel_finetune and self.use_cross_attention):  
            modified_novel_q, attn_weights_n_b_q = self.base_to_novel_cross_attention_layers(
                output[-self.num_novel_classes:], output[ 0:-self.num_novel_classes], pos=query_embed[ 0:-self.num_novel_classes ], query_pos=query_embed[-self.num_novel_classes:], novel=False
            )
            output = torch.cat((output[0:-self.num_novel_classes],modified_novel_q), dim=0)

        outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
 
        if (self.novel_finetune and self.use_mlp): 
            b,o,c,h,w = mask_features.shape[:]
            novel_output_mask  = torch.einsum("bqc,bochw->boqhw", self.novel_mask_embed, mask_features)
            outputs_mask[:,:,-self.num_novel_classes:, :, :] += novel_output_mask

        predictions_mask.append(outputs_mask.unsqueeze(0))

        layerwise_attn_weights = []
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            #if not self.use_multi_token:
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            input_level =  src[level_index]
            pos_level = pos[level_index]
            
            if self.novel_finetune:
                if b==1 and o!=1:
                    batch_val_size = self.query_feat.weight.shape[1]
                    input_level = input_level.repeat(1,batch_val_size,1)
                    pos_level = pos_level.repeat(1,batch_val_size,1)
            
            output, attn_weights = self.transformer_cross_attention_layers[i](
                output, input_level,
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos_level, query_pos=query_embed
            )
        
            layerwise_attn_weights.append(attn_weights)


            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            ###ADDED JUST NOW
            if (self.novel_finetune and self.use_cross_attention):
                modified_novel_q, attn_weights_n_b_q = self.base_to_novel_cross_attention_layers(
                    output[-self.num_novel_classes:], output[ 0:-self.num_novel_classes], pos=query_embed[ 0:-self.num_novel_classes ], query_pos=query_embed[-self.num_novel_classes:], novel=False
                )
                output = torch.cat((output[0:-self.num_novel_classes],modified_novel_q), dim=0)

            outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            
            if (self.novel_finetune and self.use_mlp):
                b,o,c,h,w = mask_features.shape[:]
                #import pdb; pdb.set_trace()
                novel_output_mask  = torch.einsum("bqc,bochw->boqhw", self.novel_mask_embed, mask_features)
                outputs_mask[:,:,-self.num_novel_classes:, :, :] += novel_output_mask
    
            predictions_mask.append(outputs_mask.unsqueeze(0))

        assert len(predictions_mask) == self.num_layers + 1

        out = {
                'pred_masks': predictions_mask[-1].squeeze(0)
            }

        return out


    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        #import pdb; pdb.set_trace()
        if mask_features.ndim == 5:
            b,o,_,_,_ = mask_features.shape
            if b==1 and o!=1:
                batch_val_size = self.query_feat.weight.shape[1]
                output = output[:,:batch_val_size,:]
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        if self.use_mlp:
            #import pdb; pdb.set_trace()
            mask_embed = self.mask_embed(decoder_output)
        else:
            mask_embed = decoder_output
        if mask_features.ndim == 5:
                
            outputs_mask  = torch.einsum("bqc,bochw->boqhw", mask_embed, mask_features)
        else:
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        
        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        if mask_features.ndim == 5:
            if (b==1 and o==1):
                attn_mask = F.interpolate(outputs_mask.squeeze(1), size=attn_mask_target_size, mode="bilinear", align_corners=False)
            elif (b==1 and o!=1):
                attn_mask = F.interpolate(outputs_mask.reshape(-1, *outputs_mask.size()[2:]), size=attn_mask_target_size, mode="bilinear", align_corners=False)
            else:
                attn_mask = F.interpolate(outputs_mask.squeeze(), size=attn_mask_target_size, mode="bilinear", align_corners=False)
        else:
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()

        attn_mask = attn_mask.detach()
        
        return  outputs_mask, attn_mask

