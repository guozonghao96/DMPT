#!/usr/bin/env bash
# echo 'sleep 3.5h'
# sleep 3.5h
export OMP_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3 #,4,5,6,7
# GPU_NUM=4 #8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPU_NUM=8


CONFIG='configs/dpm_psis/attnshift_voc12aug_1x_dpm_matching.py'
# CONFIG='configs/dpm_psis/attnshift_voc12aug_1x_dpm_matching_high_res.py'
# WORK_DIR='../work_dirs/dpm_psis/init_sk_match_dpm_voc12aug_1x_dw_1_ce_lw_0_05_1attn_3dpm_s1_hid256_map_thr_0_9_ex_iam_pos_proposal_top_30_high_res'
WORK_DIR='../work_dirs/dpm_psis/init_sk_match_dpm_voc12aug_1x_dw_1_ce_lw_0_05_1attn_3dpm_s1_hid256_map_thr_0_9_ex_iam_negloss_02_init_10_outer_5_negthr_1_pos_proposal_top_30'

python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50045 --use_env ./tools/train.py \
    ${CONFIG} \
    --cfg-options \
        model.keypoint_head.feature_based_deform=False \
        model.keypoint_head.sklearn_matching=True \
        model.keypoint_head.meanshift_refine_times=5 \
        model.keypoint_head.semantic_deform=False \
        model.keypoint_head.hidden_channels=256 \
        model.keypoint_head.deformable_cost_weight=1.0 \
        model.keypoint_head.topk_stride=1 \
        model.keypoint_head.part_points_topk=3 \
        model.keypoint_head.instance_neg_points_topk=0 \
        model.keypoint_head.map_thr=0.9 \
        model.keypoint_head.neg_loss_weight=0.02 \
        model.keypoint_head.num_classes=21 \
        model.keypoint_head.part_proposal_topk=30 \
        model.keypoint_head.ablation_cls_weight=False \
        model.keypoint_head.root_cls_weight=5 \
        model.keypoint_head.bce_loss.loss_weight=0.05 \
        model.keypoint_head.bce_loss.use_sigmoid=False \
        model.keypoint_head.mask_gt_sets=3 \
        model.keypoint_head.num_neg_sample=5 \
        model.keypoint_head.iam_num_points_init_neg=5 \
        model.keypoint_head.iam_num_points_init_pos=5 \
        model.keypoint_head.dpm_fg_thr=0.0 \
        model.keypoint_head.num_semantic_points=5 \
        model.keypoint_head.num_classifier=5 \
        model.keypoint_head.random_iam_points=False \
        model.keypoint_head.part_mean_max=False \
        model.keypoint_head.iam_pos_thr=0.0 \
        model.keypoint_head.iam_neg_thr=0.1 \
        model.keypoint_head.exclude_iam_pos=True \
        model.keypoint_head.exclude_iam_neg=True \
        data.samples_per_gpu=1 \
        data.workers_per_gpu=1 \
        optimizer.lr=0.0001 \
        optimizer_config.update_interval=2 \
    --work-dir ${WORK_DIR} \
    --auto-resume \
    --gpus ${GPU_NUM} --launcher pytorch
    
#     --resume ${WORK_DIR}/latest.pth \



# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50042 --use_env ./tools/test.py \
#     ${CONFIG} \
#     ${WORK_DIR}/epoch_10.pth \
#     --eval mAP_Segm --launcher pytorch

    # --out pseudo_gt_preds.pkl --launcher pytorch
    
    