import torch
import pickle
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pylab
import json
import pickle
from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

from mmcv.runner.checkpoint import load_checkpoint
from mmcv import Config, DictAction
from mmdet.datasets import (build_dataloader, build_dataset,
                          replace_ImageToTensor)

cfg = Config.fromfile('../work_dirs/lsl/lsl_voc0712p_1x_only_mil/lsl_voc0712p_1x_mil_simple_fpn_fcos_refinment.py')
datasets = [build_dataset(cfg.data.test)]

import pickle 
results = pickle.load(open('pseudo_gt_preds.pkl', 'rb'))
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
iou_s = []
iou_m = []
iou_l = []
ious = []

for result_per_img in results:
    for det_per_cls in result_per_img:
        for det in det_per_cls:
            bbox = torch.from_numpy(np.array(det[:4], dtype=np.int)).reshape(1, -1)
            score = np.array(det[4])
            gt = torch.from_numpy(np.array(det[5:], dtype=np.int)).reshape(1, -1)
            iou = bbox_overlaps(bbox, gt, is_aligned=True)
            ious.append(iou)
            
            scale = (gt[0][2] - gt[0][0]) * (gt[0][3] - gt[0][1])
            if scale <= 32 * 32:
                iou_s.append(iou)
            elif scale > 32 * 32 and scale <= 96 * 96:
                iou_m.append(iou)
            elif scale > 96 * 96:
                iou_l.append(iou)
                
miou = sum(ious) / len(ious)
miou_s = sum(iou_s) / len(iou_s)
miou_m = sum(iou_m) / len(iou_m)
miou_l = sum(iou_l) / len(iou_l)
print(miou, miou_s, miou_m, miou_l)