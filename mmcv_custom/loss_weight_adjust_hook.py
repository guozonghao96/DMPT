import mmcv
from mmcv.runner import Hook, HOOKS

@HOOKS.register_module()
class LossWeightAdjustHook(Hook):
    def __init__(self, start_epoch=1, 
                 decoder_cls_loss_weight=1.0,
                 decoder_bbox_loss_weight=10.0,
                 decoder_mask_loss_weight=1.0,
                 **kwargs):
        self.start_epoch = start_epoch
        self.decoder_cls_loss_weight = decoder_cls_loss_weight
        self.decoder_bbox_loss_weight = decoder_bbox_loss_weight
        self.decoder_mask_loss_weight = decoder_mask_loss_weight

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        begin_flag = epoch > self.start_epoch
        runner.model.module.roi_head.bbox_head.loss_cls.loss_weight = begin_flag * self.decoder_cls_loss_weight
        runner.model.module.roi_head.bbox_head.loss_bbox.loss_weight = begin_flag * self.decoder_bbox_loss_weight
        runner.model.module.roi_head.mask_head.loss_mask.loss_weight = begin_flag * self.decoder_mask_loss_weight