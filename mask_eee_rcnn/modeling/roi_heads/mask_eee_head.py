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


def mask_eee_loss(pred_mask_eee, true_positive_mask, true_negative_mask, false_positive_mask, false_negative_mask, loss_weight):
    """
    Compute the maskiou loss.

    Args:
        labels (Tensor): Given mask labels
        pred_maskiou: Predicted maskiou
        gt_maskiou: Ground Truth IOU generated in mask head
    """
    gt_mask = torch.cat([
            true_positive_mask.unsqueeze(1),
            true_negative_mask.unsqueeze(1),
            false_positive_mask.unsqueeze(1),
            false_negative_mask.unsqueeze(1),
        ], dim=1).to(dtype=torch.float) # [N, 4, H, W]
    gt_mask = torch.argmax(gt_mask, dim=1).to(dtype=torch.long) # [N, 4, H, W] = to [N, H, W]
    loss = F.cross_entropy(
            pred_mask_eee, 
            gt_mask,
            reduction='mean')
    # import cv2
    # import numpy as np
    # n_masks = pred_mask_eee.shape[0]
    # for idx in range(n_masks):
    #     pred = pred_mask_eee[idx]
    #     pred = torch.argmax(pred, dim=0, keepdim=True)
    #     pred = torch.cat([pred == i for i in range(4)], dim=0) # ignore false negative
    #     pred = pred.detach().cpu().numpy().astype(np.uint8) * 255
    #     pred = pred.transpose(1, 2, 0) # [3, H, W] -> [H, W, 3]
    #     pred_vis = np.zeros([pred.shape[0], pred.shape[1], 3])
    #     pred_vis[:, :, 0] = pred[:, :, 1]
    #     pred_vis[:, :, 1] = pred[:, :, 0]
    #     pred_vis[:, :, 2] = pred[:, :, 2]
    #     gt_vis = np.zeros([pred.shape[0], pred.shape[1], 3])
    #     tp_mask = true_positive_mask[idx].unsqueeze(0).detach().cpu().numpy()
    #     tp_mask = (tp_mask * 255).astype(np.uint8).transpose(1, 2, 0)
    #     tn_mask = true_negative_mask[idx].unsqueeze(0).detach().cpu().numpy()
    #     tn_mask = (tn_mask * 255).astype(np.uint8).transpose(1, 2, 0)
    #     fp_mask = false_positive_mask[idx].unsqueeze(0).detach().cpu().numpy()
    #     fp_mask = (fp_mask * 255).astype(np.uint8).transpose(1, 2, 0)
    #     gt_vis[:, :, 0] = tn_mask[:, :, 0]
    #     gt_vis[:, :, 1] = tp_mask[:, :, 0]
    #     gt_vis[:, :, 2] = fp_mask[:, :, 0]
    #     cv2.imwrite('mask_{}.png'.format(idx), np.hstack([pred_vis, gt_vis]))

    # return {'loss_mask_eee': loss * loss_weight}
    return {'loss_mask_eee': loss * loss_weight } 


def mask_eee_inference(pred_instances, pred_mask_eee):

    # pred_mask = [N, 1, H, W]
    pred_mask = torch.argmax(pred_mask_eee, dim=1, keepdim=True) # [N, 1, H, W]
    pred_mask = torch.cat([pred_mask == i for i in range(4)], dim=1)[:, :3, :, :] 
    # ignore false negative
    num_boxes_per_image = [len(i) for i in pred_instances]
    pred_mask = pred_mask.split(num_boxes_per_image, dim=0)
    for prob, instance in zip(pred_mask, pred_instances):
        instance.set('pred_error', prob)


@ROI_MASK_EEE_HEAD_REGISTRY.register()
class MaskEEEHead(nn.Module):
    def __init__(self, cfg):
        super(MaskEEEHead, self).__init__()

        input_channels = 257
        conv_dims  = cfg.MODEL.ROI_MASK_EEE_HEAD.CONV_DIM
        num_conv  = cfg.MODEL.ROI_MASK_EEE_HEAD.NUM_CONV
        self.norm         = cfg.MODEL.ROI_MASK_EEE_HEAD.NORM
        self.conv_norm_relus = []
        self.maxpool_on = cfg.MODEL.ROI_MASK_EEE_HEAD.MAXPOOL_ON

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_eee_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.add_module("mask_eee_deconv", self.deconv)

        self.predictor = Conv2d(
            conv_dims,
            4,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.add_module("mask_eee_predictor", self.predictor)

        # init weights
        for layer in self.conv_norm_relus + [self.deconv,]:
            weight_init.c2_msra_fill(layer)
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x, mask):
        mask_pool = F.max_pool2d(mask, kernel_size=2, stride=2)
        x = torch.cat((x, mask_pool), 1)
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)


def build_mask_eee_head(cfg):
    """
    Build a mask eee head defined by `cfg.MODEL.ROI_MASKIOU_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_EEE_HEAD.NAME
    return ROI_MASK_EEE_HEAD_REGISTRY.get(name)(cfg)
