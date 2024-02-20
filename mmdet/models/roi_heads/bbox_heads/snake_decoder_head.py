import torch.nn as nn
from .snake import Snake
from .utils import prepare_training_evolve, img_poly_to_can_poly, get_gcn_feature, get_adj_ind
import torch
from .snake import Snake
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import auto_fp16, force_fp32

@HEADS.register_module()
class SnakeDecoderHead(nn.Module):
    def __init__(self, 
                 # in_channel, 
                 state_dim=256, 
                 feature_dim=128, 
                 conv_type='dgrid',
                 snake_config=dict(
                     ro=4,
                     adj_num=4),
                loss_energy=dict(
                    type='SnakeEnergyLoss',
                    alpha=1.0,
                    # beta=0.005,
                    gamma=0.0,
                    # sigma=0.001,
                    reduction='mean', 
                    loss_weight=0.01)
                ):
        super(SnakeDecoderHead, self).__init__()
        self.evolve_gcn = Snake(state_dim, 
                              feature_dim=feature_dim + 2, 
                              conv_type=conv_type)
        self.snake_config = snake_config
        self.loss_energy = build_loss(loss_energy)
        
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        #         m.weight.data.normal_(0.0, 0.01)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
    def init_weights(self):
        self.apply(self._init_weights)
        
    def prepare_training(self, batch):
        init = prepare_training_evolve(batch)
        return init
    
        # i_it_4py = batch['i_it_4py'] # 检测框生成的 4个中心点
        # c_it_4py = batch['c_it_4py'] # i_it_4py 减去水平框xmin ymin产生的 水平不变的框
        # i_gt_4py = batch['i_gt_4py'] # gt的mask 4 个角点
        # c_gt_4py = batch['c_gt_4py'] # i_gt_4py - xmin ymin之后水平不变的点
        
        # i_it_py = batch['i_it_py']  # 检测框生成的 128个采样点的多边形
        # c_it_py = batch['c_it_py']  # i_it_py 减去水平框xmin ymin产生的 水平不变的框
        # i_gt_py = batch['i_gt_py']  # gt的mask的边界采样点
        # c_gt_py = batch['c_gt_py']  # gt的mask的边界采样点 - xmin ymin之后水平不变的点

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        c_it_poly = c_it_poly * self.snake_config.ro
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        adj = get_adj_ind(self.snake_config.adj_num, init_input.size(2), init_input.device)
        i_poly = i_it_poly * self.snake_config.ro + snake(init_input, adj).permute(0, 2, 1)
        return i_poly

    # def forward(self, cnn_feature, snakes, snakes_targets):
    def forward(self, cnn_feature, snakes):
        num_points = snakes[0].size(1)
        batch_size = cnn_feature.size(0)
        device = cnn_feature.device
        
        with torch.no_grad():
            ct_num = torch.as_tensor([len(snake) for snake in snakes]).to(device)
            
            ct_01 = torch.zeros((batch_size, max(ct_num))).to(device)
            for i_batch, num in enumerate(ct_num):
                ct_01[i_batch][:num] = 1
                
            # 我们的框架里面所有的点都是原图大小，而deep snake里面是下采样4倍大小, 因此需要除以4获得相同的
            # / 4是在 位移场的cnn map上做的， / 16是在vit的特征上做的
            i_it_py = torch.zeros((batch_size, max(ct_num), num_points, 2)).to(device) 
            # i_gt_py = torch.zeros((batch_size, max(ct_num), -1, 2)).to(device) / self.snake_config.ro #这个targets要均匀采样的
            # for i_batch, (num, snake, snake_target) in enumerate(zip(ct_num, snakes, snakes_targets)):
            for i_batch, (num, snake) in enumerate(zip(ct_num, snakes)):
                i_it_py[i_batch, :num] = snake
                # i_gt_py[i_batch, :num] = snake_target
            i_it_py /= self.snake_config.ro
            c_it_py = img_poly_to_can_poly(i_it_py)
            # c_gt_py = img_poly_to_can_poly(i_gt_py) #这个targets要均匀采样的
            
            batch = dict(
                ct_01=ct_01,
                i_it_py=i_it_py,
                # i_gt_py=i_gt_py,
                c_it_py=c_it_py,
                # c_gt_py=c_gt_py,
                meta=dict(ct_num=ct_num)
            )
            init = self.prepare_training(batch)
            
        py_pred = self.evolve_poly(self.evolve_gcn, cnn_feature, init['i_it_py'], init['c_it_py'], init['py_ind'])
        # py_preds = [py_pred]
        # for i in range(self.iter):
        #     py_pred = py_pred / snake_config.ro
        #     c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred)
        #     evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
        #     py_pred = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred, init['py_ind'])
        #     py_preds.append(py_pred)
        # ret.update({'py_pred': py_preds, 'i_gt_py': output['i_gt_py'] * snake_config.ro})
        output = dict(
            py_pred=py_pred,
            # i_gt_py=init['i_gt_py'] * self.snake_config.ro
        )
        return output

    @force_fp32(apply_to=('decoded_snakes'))
    def loss(self,
             decoded_snakes,
             pred_offset_map,
             gt_points,
             snake_targets,
             snake_weights
            ):
        losses = dict()        
        pred_offset_map = pred_offset_map.detach() # 应该是detach掉梯度谱 [batch, 2, h, w]
        loss_energy = 0
        for snakes, energy_map, points, snakes_t, weight in zip(decoded_snakes, 
                                                        pred_offset_map, 
                                                        gt_points,
                                                        snake_targets,
                                                        snake_weights
                                                       ):
            loss_energy += self.loss_energy(snakes, 
                                            energy_map,
                                            points,
                                            snakes_t,
                                            weight
                                           )
        losses['loss_energy'] = loss_energy
        return losses

