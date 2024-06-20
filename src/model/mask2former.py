import torch
import torch.nn.functional as F
from torch import nn

from .resnet import resnet50, resnet101
from .multi_scale_transformer import MultiScaleMaskedTransformerDecoder

from .msdeformattn import MSDeformAttnPixelDecoder
import torch.nn.functional as F



def get_model_m2former(args, novel_fine_tune=False) -> nn.Module:
    #return PSPNet(args, zoom_factor=8, use_ppm=True)
    
    return Mask2former(args, novel_fine_tune)



class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            # self.features.append(nn.Sequential(
            #     nn.AdaptiveAvgPool2d(bin),
            #     #nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            #     #nn.BatchNorm2d(reduction_dim),
            #     nn.BatchNorm2d(in_dim),
            #     nn.ReLU(inplace=True)))
            
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)))

        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        
        #import pdb; pdb.set_trace()
        return torch.cat(out, 1)


class Mask2former(nn.Module):
    def __init__(self, args, novel_fine_tune=False):
        super(Mask2former, self).__init__()
        
        if args.get('num_classes_tr') is None:
            args.num_classes_tr = args.classes
        if args.layers == 50:
            resnet = resnet50(pretrained=args.pretrained)
        else:
            resnet = resnet101(pretrained=args.pretrained)
        
        #import pdb; pdb.set_trace()
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.feature_res = (105, 105)
        self.avgpool = resnet.avgpool
        self.get_last_layer = True

        fea_dim = 2048

        self.ppm = PPM(fea_dim, int(fea_dim/len(args.bins)), args.bins)
        fea_dim *= 2
        #fea_dim *= 5
        self.bottleneck = nn.Sequential(
           nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
           nn.BatchNorm2d(512),
           nn.ReLU(inplace=True),
           #nn.Dropout2d(p=args.dropout),
        )
        
        # #NOTE: USUALLY THE KERNEL SIZE IS 3
        self.projection_output = nn.Sequential(
           nn.Conv2d(512+256, 256, kernel_size=3, padding=1, bias=False),
           nn.BatchNorm2d(256),
           nn.ReLU(inplace=True),
           nn.Dropout2d(p=args.dropout),
        )

        self.pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape=[256, 512, 1024, 2048], 
            input_stride= [4,8,16,32],
            conv_dim = 256,
            mask_dim = 256)

        if novel_fine_tune:
            num_classes = args.num_classes_tr + args.num_novel
            if args.one_vs_rest:
                #import pdb; pdb.set_trace()
                num_classes = args.num_classes_tr + 1
                args.num_novel=1
        else:
            num_classes = args.num_classes_tr 

        self.keep_class_embed = args.keep_class_embed

        self.transformer = MultiScaleMaskedTransformerDecoder(in_channels=256, mask_classification=self.keep_class_embed, num_classes = num_classes, hidden_dim=256, num_queries=num_classes, nheads=8, dim_feedforward=2048, dec_layers= 9, pre_norm=False, mask_dim=256, num_novel=args.num_novel, novel_finetune=novel_fine_tune, use_mlp=args.use_mlp, num_base_classes=args.num_classes_tr, shots=args.shot)


    def extract_features(self, x):
        i_h = x.shape[2]
        i_w = x.shape[3]
        x = self.layer0(x)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        pixel_decoder_input = {'c1':x_1, 'c2':x_2, 'c3':x_3, 'c4':x_4}

        mask_features, multi_scale_features = self.pixel_decoder(pixel_decoder_input)
        x = self.ppm(x_4)
        x = self.bottleneck(x)
        bottleneck_output = x

        x = F.interpolate(x, size=(mask_features.shape[-2], mask_features.shape[-1]), mode="bilinear")
        x = torch.cat((x,mask_features), dim=1)

        mask_features = self.projection_output(x)

        #import pdb; pdb.set_trace()
        
        if self.get_last_layer:
            
            return mask_features, multi_scale_features, bottleneck_output
        else:
            return mask_features, multi_scale_features, None


    def forward(self, x, mask_loss=False):
        #with torch.no_grad():
        i_h = x.shape[2]
        i_w = x.shape[3]
        x = self.layer0(x)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        pixel_decoder_input = {'c1':x_1, 'c2':x_2, 'c3':x_3, 'c4':x_4}
        mask_features, multi_scale_features = self.pixel_decoder(pixel_decoder_input)
        
        ####NOTE: ADDING PSPNET
        x = self.ppm(x_4)
       
        x = self.bottleneck(x)
        x = F.interpolate(x, size=(mask_features.shape[-2], mask_features.shape[-1]), mode="bilinear")
        x = torch.cat((x,mask_features), dim=1)

        mask_features = self.projection_output(x)
       
        out = self.transformer(multi_scale_features, mask_features)
        
        mask_features = out['pred_masks']
        ndims = mask_features.dim()
        if ndims==3:
            mask_features = mask_features.unsqueeze(0)
        mask_features_aux =  out['aux_masks']
        if mask_features_aux.dim() <5:
            mask_features_aux = mask_features_aux.unsqueeze(0)
        mask_features_aux = mask_features_aux.permute(1,0,2,3,4)
        mask_cls = out['pred_logits']
        mask_cls_aux = out['aux_logits']
        #layer_wise_attention = out['layerwise_attention']
        if self.keep_class_embed:
            if mask_cls.dim() <3:
                    mask_cls = mask_cls.unsqueeze(0)   
            if mask_cls_aux.dim() <4:
                mask_cls_aux = mask_cls_aux.unsqueeze(0)
            mask_cls_aux = mask_cls_aux.permute(1,0,2,3)

        if not mask_loss:
            mask_features = F.interpolate(mask_features, size=(i_h, i_w), mode="bilinear")
            semseg = mask_features
            if self.keep_class_embed:
                mask_features = mask_features.sigmoid()
                mask_cls = F.softmax(out['pred_logits'], dim=-1)
                try:
                    semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_features)
                except:
                    semseg =  torch.einsum("bqc,bqhw->bchw", mask_cls.unsqueeze(0), mask_features)

            semseg_aux = mask_features_aux
            if self.keep_class_embed:
                mask_features_aux = mask_features_aux.sigmoid()
                mask_cls_aux = F.softmax(mask_cls_aux, dim=-1)
                try:
                    semseg_aux = torch.einsum("blqc,blqhw->blchw", mask_cls_aux, mask_features_aux)
                except:
                    semseg_aux = torch.einsum("blqc,blqhw->blchw", mask_cls_aux.unsqueeze(0), mask_features_aux)

            return semseg, semseg_aux
        else:
            out = {
                'pred_logits': mask_cls,
                'pred_masks': mask_features,
                'aux_masks':mask_features_aux,
                'aux_logits':mask_cls_aux
                
            }
            return out
