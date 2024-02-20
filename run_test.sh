#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
GPU_NUM=8

CONFIG='configs/dpm_psis/attnshift_voc12aug_1x_dpm_matching.py'
# CONFIG='configs/dpm_psis/attnshift_voc12aug_1x_dpm_matching_sam.py'
# WORK_DIR='./DMPT_VOC2012_SOTA_LOG'    
WORK_DIR='../work_dirs/dpm_psis/init_sk_match_dpm_voc12aug_1x_dw_1_ce_lw_0_05_1attn_3dpm_s1_hid256_map_thr_0_9_ex_iam_negloss_02_init_10_outer_5_negthr_1_pos_proposal_top_30'

python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50042 --use_env ./tools/test.py \
    ${CONFIG} \
    ${WORK_DIR}/epoch_1.pth \
    --cfg-options \
        model.test_wo_detector=False \
    --eval mAP_Segm --out gt_masks.pkl --launcher pytorch
    
# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50042 --use_env ./tools/test.py \
#     ${CONFIG} \
#     ${WORK_DIR}/epoch_12.pth \
#     --cfg-options \
#         model.test_wo_detector=Fasle \
#     --eval mAP_Segm --out pseudo_gt_preds.pkl --launcher pytorch
#     # --out pseudo_gt_preds.pkl --launcher pytorch