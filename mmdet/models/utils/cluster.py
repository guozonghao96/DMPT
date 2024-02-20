import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PMMs(nn.Module):

    def __init__(self, c, k=3, stage_num=10, kappa=20):
        super(PMMs, self).__init__()
        self.stage_num = stage_num
        self.num_pro = k
        mu = torch.Tensor(1, c, k).cuda()
        mu.normal_(0, math.sqrt(2. / k))  # Init mu
        self.mu = self._l2norm(mu, dim=1)
        self.kappa = kappa
        #self.register_buffer('mu', mu)
        
    def forward(self, feature, fg_mask, bboxes):
        # prototypes, mu_f, mu_b = self.generate_prototype(feature, fg_mask, bboxes)
        fg_prototypes, fg_feature_maps = self.generate_prototype(feature, fg_mask, bboxes)
        part_attn_maps, semantic_points, visible_weights = self.get_points(fg_feature_maps, 
                                                                           fg_prototypes, 
                                                                           fg_mask, bboxes)

        return part_attn_maps, semantic_points, visible_weights

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def EM(self,x):
        '''
        EM method
        :param x: feauture  b * c * n
        :return: mu
        '''
        b = x.shape[0]
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                # E STEP:
                z = self.Kernel(x, mu)
                z = F.softmax(z, dim=2)  # b * n * k
                # M STEP:
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k

                mu = self._l2norm(mu, dim=1)

        mu = mu.permute(0, 2, 1)  # b * k * c

        return mu

    def Kernel(self, x, mu):
        x_t = x.permute(0, 2, 1)  # b * n * c
        z = self.kappa * torch.bmm(x_t, mu)  # b * n * k

        return z

    def get_prototype(self,x):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        mu = self.EM(x) # b * k * c

        return mu

    def generate_prototype(self, feature, mask, bboxes):
        patch_h, patch_w = feature.size()[-2:]
        mask = mask.bool().float()
        mask_inter = F.interpolate(mask.unsqueeze(1), (patch_h, patch_w), mode='bilinear')

        # foreground
        z = mask_inter * feature.unsqueeze(0)
        mu_f = self.get_prototype(z)
        
        return mu_f, z

#         # background
#         print(bboxes.size())
#         z = mask_inter * feature.unsqueeze(0)
#         print()
#         mu_f = self.get_prototype(z)
        
        
        # mask_bg = 1-mask
        # background
        # z_bg = mask_bg * feature
        # mu_b = self.get_prototype(z_bg)

        # return mu_, mu_f, mu_b

    def get_points(self, feature_maps, prototypes, fg_mask, bboxes):
        # num_gt, c, h, w
        # num_gt, 5, c
        num_gt, c, patch_h, patch_w = feature_maps.size()
        feat = feature_maps.flatten(2)
        sim_matrix = torch.bmm(prototypes, feat)
        sim_matrix = F.interpolate(sim_matrix.reshape(num_gt, self.num_pro, patch_h, patch_w), 
                                   (patch_h * 16, patch_w * 16), mode='bilinear') # num_gt, num_pro, img_H, img_W
        max_args = (sim_matrix.max(1)[1] + 1) * fg_mask.bool() # num_gt, img_H, img_W
        
        # for循环获得visible和invisible的位置信息和响应
        
        part_attn_maps = []
        semantic_points = []
        visible_weights = []
        for index_mask, attn_maps in zip(max_args, sim_matrix):
            for i in range(self.num_pro):
                attn = attn_maps[i]
                assign = (index_mask == i + 1)
                attn[~assign] = 0
                # attn 就是当前part的响应图，除了被分配到位置的value为原来数值，其他位置都为0
                part_attn_maps.append(attn)
                
                # 判断是否是visible的
                if assign.sum() == 0: # 没有响应，说明这个part是invisible的
                    visible_weights.append(torch.zeros(1).long().to(max_args.device))
                    semantic_points.append(-1e-3 * torch.ones(1, 2).to(max_args.device))
                else:
                    visible_weights.append(torch.ones(1).long().to(max_args.device))
                    max_value = attn.max() # 可能存在多个max极值，但是我们只取一个
                    max_coord = (attn == max_value).nonzero().reshape(-1, 2).flip(-1)[:1]
                    semantic_points.append(max_coord)
        part_attn_maps = torch.cat(part_attn_maps).reshape(num_gt, self.num_pro, patch_h * 16, patch_w * 16)  
        semantic_points = torch.cat(semantic_points).reshape(num_gt, self.num_pro, 2)  
        visible_weights = torch.cat(visible_weights).reshape(num_gt, self.num_pro)
        return part_attn_maps, semantic_points, visible_weights
    
#     def discriminative_model(self, query_feature, mu_f, mu_b):

#         mu = torch.cat([mu_f, mu_b], dim=1)
#         mu = mu.permute(0, 2, 1)

#         b, c, h, w = query_feature.size()
#         x = query_feature.view(b, c, h * w)  # b * c * n
#         with torch.no_grad():

#             x_t = x.permute(0, 2, 1)  # b * n * c
#             z = torch.bmm(x_t, mu)  # b * n * k

#             z = F.softmax(z, dim=2)  # b * n * k

#         P = z.permute(0, 2, 1)

#         P = P.view(b, self.num_pro * 2, h, w) #  b * k * w * h  probability map
#         P_f = torch.sum(P[:, 0:self.num_pro], dim=1).unsqueeze(dim=1) # foreground
#         P_b = torch.sum(P[:, self.num_pro:], dim=1).unsqueeze(dim=1) # background

#         Prob_map = torch.cat([P_b, P_f], dim=1)

#         return Prob_map, P