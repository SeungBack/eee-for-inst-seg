# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
import pycocotools.mask as mask_utils

from mask_eee_rcnn.layers import Conv2d, ShapeSpec, cat, get_norm, ConvTranspose2d
from mask_eee_rcnn.utils.events import get_event_storage
from mask_eee_rcnn.utils.registry import Registry

ROI_MASK_EEE_HEAD_REGISTRY = Registry("ROI_MASK_EEE_HEAD")
ROI_MASK_EEE_HEAD_REGISTRY.__doc__ = """
Registry for maskiou heads, which predicts predicted mask iou.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def mask_eee_loss(pred_mask_eee, true_positive_mask, false_positive_mask, false_negative_mask, loss_weight):
    """
    Compute the maskiou loss.

    Args:
        labels (Tensor): Given mask labels
        pred_maskiou: Predicted maskiou
        gt_maskiou: Ground Truth IOU generated in mask head
    """
    losses = {}
    for target, pred_mask in pred_mask_eee.items():
        pred_mask = pred_mask.squeeze(1)
        if target == 'TP':
            loss = F.binary_cross_entropy_with_logits(
                pred_mask, true_positive_mask.to(dtype=torch.float)
                , reduction='mean')
        elif target == 'FP':
            loss = F.binary_cross_entropy_with_logits(
                pred_mask, false_positive_mask.to(dtype=torch.float)
                , reduction='mean')
        elif target == 'FN':
            loss = F.binary_cross_entropy_with_logits(
                pred_mask, false_negative_mask.to(dtype=torch.float)
                , reduction='mean')
        losses['loss_mask_eee_' + target] = loss * loss_weight
    return losses


def mask_eee_inference(pred_instances, pred_mask_eee):

    for target, pred_mask in pred_mask_eee.items():
        # pred_mask = [N, 1, H, W]
        pred_mask = pred_mask.unsqueeze(1).sigmoid().squeeze(1)
        num_boxes_per_image = [len(i) for i in pred_instances]
        pred_mask = pred_mask.split(num_boxes_per_image, dim=0)
        for prob, instance in zip(pred_mask, pred_instances):
            instance.set('pred_' + target, prob)


@ROI_MASK_EEE_HEAD_REGISTRY.register()
class MaskEEEHead(nn.Module):
    def __init__(self, cfg):
        super(MaskEEEHead, self).__init__()

        input_channels = 257
        conv_dims_shared  = cfg.MODEL.ROI_MASK_EEE_HEAD.CONV_DIM_SHARED
        conv_dims_target  = cfg.MODEL.ROI_MASK_EEE_HEAD.CONV_DIM_TARGET
        num_conv_shared  = cfg.MODEL.ROI_MASK_EEE_HEAD.NUM_CONV_SHARED
        num_conv_target  = cfg.MODEL.ROI_MASK_EEE_HEAD.NUM_CONV_TARGET
        self.num_conv_target = num_conv_target
        self.norm         = cfg.MODEL.ROI_MASK_EEE_HEAD.NORM
        targets           = cfg.MODEL.ROI_MASK_EEE_HEAD.TARGETS

        self.conv_norm_relus_shared = []

        for k in range(num_conv_shared):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims_shared,
                conv_dims_shared,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims_shared),
                activation=F.relu,
            )
            self.add_module("mask_eee_fcn{}".format(k + 1), conv)
            self.conv_norm_relus_shared.append(conv)

        self.conv_norm_relus_targets = {}
        for target in targets:
            self.conv_norm_relus_targets[target] = []
            for k in range(num_conv_target):
                conv = Conv2d(
                    conv_dims_shared if k == 0 else conv_dims_target,
                    conv_dims_target,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not self.norm,
                    norm=get_norm(self.norm, conv_dims_target),
                    activation=F.relu,
                )
                self.add_module("mask_eee_fcn{}_{}".format(k + 1, target), conv)
                self.conv_norm_relus_targets[target].append(conv)

        self.deconvs = {}        
        for target in targets:
            deconv = ConvTranspose2d(
                conv_dims_target,
                conv_dims_target,
                kernel_size=2,
                stride=2,
                padding=0,
            )
            self.add_module("mask_eee_deconv_{}".format(target), deconv)
            self.deconvs[target] = deconv

        self.predictors = {}
        for target in targets:
            predictor = Conv2d(
                conv_dims_target,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            self.add_module("mask_eee_predictor_{}".format(target), predictor)
            self.predictors[target] = predictor

        # init weights
        for layer in self.conv_norm_relus_shared:
            weight_init.c2_msra_fill(layer)
        for target, layer_list in self.conv_norm_relus_targets.items():
            for layer in layer_list:
                weight_init.c2_msra_fill(layer)
        for _, layer in self.deconvs.items():
            weight_init.c2_msra_fill(layer)
        for _, layer in self.predictors.items():
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.constant_(layer.bias, 0)


    def forward(self, x, mask):
        mask_pool = F.max_pool2d(mask, kernel_size=2, stride=2)
        x_shared = torch.cat((x, mask_pool), 1)
        for layer in self.conv_norm_relus_shared:
            x_shared = layer(x_shared)
        pred_masks = {}
        for target, layers in self.conv_norm_relus_targets.items():
            for i, layer in enumerate(layers):
                if i == 0:
                    x = layer(x_shared)
                else:
                    x = layer(x)
                if i == self.num_conv_target - 1:
                    x = F.relu(self.deconvs[target](x))
                    x = self.predictors[target](x)
                    pred_masks[target] = x
        return pred_masks


def build_mask_eee_head(cfg):
    """
    Build a mask eee head defined by `cfg.MODEL.ROI_MASKIOU_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_EEE_HEAD.NAME
    return ROI_MASK_EEE_HEAD_REGISTRY.get(name)(cfg)
