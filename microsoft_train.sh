#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4 #0 #,1,2,3
GPU_NUM=1 #4 #8

CONFIG='configs/dpm_psis/attnshift_voc12aug_1x_dpm_matching.py'
# WORK_DIR='../work_dirs/dpm_psis/deformable_part_matching_voc12aug_1x_dw_1_sep_softmax_ce_loss_lw_0_05_attn_plus_top7_dpm_point'
WORK_DIR='../work_dirs/dpm_psis/debug'

python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50055 --use_env ./tools/train.py \
    ${CONFIG} \
    --cfg-options \
        model.keypoint_head.semantic_deform=False \
        model.keypoint_head.hidden_channels=256 \
        model.keypoint_head.deformable_cost_weight=1.0 \
        model.keypoint_head.part_points_topk=3 \
        model.keypoint_head.map_thr=0.9 \
        model.keypoint_head.neg_loss_weight=0.0 \
        model.keypoint_head.num_classes=21 \
        model.keypoint_head.bce_loss.loss_weight=0.05 \
        model.keypoint_head.bce_loss.use_sigmoid=False \
        model.keypoint_head.mask_gt_sets=3 \
        model.keypoint_head.num_semantic_points=6 \
        model.keypoint_head.num_classifier=6 \
        data.samples_per_gpu=1 \
        data.workers_per_gpu=1 \
        optimizer.lr=0.0001 \
        optimizer_config.update_interval=4 \
    --work-dir ${WORK_DIR} \
    --auto-resume \
    --gpus ${GPU_NUM} --launcher pytorch

#     --resume ${WORK_DIR}/latest.pth \
    

# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50042 --use_env ./tools/test.py \
#     ${CONFIG} \
#     ${WORK_DIR}/latest.pth \
#     --eval mAP_Segm --launcher pytorch

#     --out pseudo_gt_preds.pkl --launcher pytorch
    
    

