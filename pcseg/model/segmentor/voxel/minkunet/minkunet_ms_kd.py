'''
Reference:
    [1] https://github.com/NVIDIA/MinkowskiEngine
    [2] https://github.com/mit-han-lab/spvnas
'''


import torchsparse
import torchsparse.nn as spnn
import torch
from torch import nn
from torchsparse import PointTensor
from pcseg.model.segmentor.base_segmentors import BaseSegmentor
from torchsparse import SparseTensor
from torchsparse.nn.utils import fapply
from .utils import initial_voxelize, voxel_to_point
from pcseg.loss import Losses
import os
import torch.nn.functional as F
import math
import copy
import numpy as np

__all__ = ['MinkUNetMsKd']


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_s, y_t):
        loss = []
        for y_s_, y_t_ in zip(y_s, y_t):
            loss.append(F.mse_loss(y_s_, y_t_))
        return loss

class SyncBatchNorm(nn.SyncBatchNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)
        
class BatchNorm(nn.BatchNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)

class BasicConvolutionBlock(nn.Module):
    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                stride=stride,
                transposed=True,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(
                outc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=1,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
        )
        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    inc, outc * self.expansion,
                    kernel_size=1,
                    dilation=1,
                    stride=stride,
                ),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=1,
                bias=False,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.Conv3d(
                outc, outc,
                kernel_size=ks,
                stride=stride,
                bias=False,
                dilation=dilation,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.Conv3d(
                outc, outc * self.expansion,
                kernel_size=1,
                bias=False,
            ),
            SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
        )
        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    inc, outc * self.expansion,
                    kernel_size=1,
                    dilation=1,
                    stride=stride,
                ),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class MinkUNetMsKd(BaseSegmentor):    
    def __init__(
        self,
        model_cfgs,
        num_class: int,
    ):
        super().__init__(model_cfgs, num_class)
        self.in_feature_dim = model_cfgs.IN_FEATURE_DIM
        self.num_layer = model_cfgs.get('NUM_LAYER', [2, 3, 4, 6, 2, 2, 2, 2])
        self.block = {
            'ResBlock': ResidualBlock,
            'Bottleneck': Bottleneck,
        }[model_cfgs.get('BLOCK', 'Bottleneck')]

        cr = model_cfgs.get('cr', 1.0)
        cs = model_cfgs.get('PLANES', [32, 32, 64, 128, 256, 256, 128, 96, 96])
        cs = [int(cr * x) for x in cs]

        self.pres = model_cfgs.get('pres', 0.05)
        self.vres = model_cfgs.get('vres', 0.05)
    
        self.sampling_type = model_cfgs.get('SAMPLING_TYPE', 'uncertain')
        self.max_voxel = model_cfgs.get('MAX_VOXEL', 3000)

        self.feat_kd = model_cfgs.get('FEAT_KD', 'mse')
        if self.feat_kd == 'mse':
            self.criterion_feat_kd = MSELoss()
        self.feat_kd_weight = model_cfgs.get('FEAT_KD_WEIGHT', 1.0)

        self.model_cfgs = model_cfgs

        self.stem_gt = nn.Sequential(
            spnn.Conv3d(
                self.in_feature_dim, cs[0],
                kernel_size=3,
                stride=1,
            ),
            SyncBatchNorm(cs[0]) if model_cfgs.IF_DIST else BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(
                cs[0], cs[0],
                kernel_size=3,
                stride=1,
            ),
            SyncBatchNorm(cs[0]) if model_cfgs.IF_DIST else BatchNorm(cs[0]),
            spnn.ReLU(True),
        )

        self.in_channels = cs[0]
        if_dist = model_cfgs.IF_DIST

        self.stage1_gt = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=model_cfgs.IF_DIST,
            ),
            *self._make_layer(
                self.block, cs[1], self.num_layer[0], if_dist=if_dist),
        )
        self.stage2_gt = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=model_cfgs.IF_DIST,
            ),
            *self._make_layer(
                self.block, cs[2], self.num_layer[1], if_dist=if_dist),
        )
        self.stage3_gt = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=model_cfgs.IF_DIST,
            ),
            *self._make_layer(
                self.block, cs[3], self.num_layer[2], if_dist=if_dist),
        )
        self.stage4_gt = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=model_cfgs.IF_DIST,
            ),
            *self._make_layer(
                self.block, cs[4], self.num_layer[3], if_dist=if_dist),
        )

        self.up1_gt = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[5],
                ks=2,
                stride=2,
                if_dist=model_cfgs.IF_DIST,
            )
        ]
        self.in_channels = cs[5] + cs[3] * self.block.expansion
        self.up1_gt.append(
            nn.Sequential(*self._make_layer(
                self.block, cs[5], self.num_layer[4], if_dist=if_dist))
        )
        self.up1_gt = nn.ModuleList(self.up1_gt)

        self.up2_gt = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[6],
                ks=2,
                stride=2,
                if_dist=model_cfgs.IF_DIST,
            )
        ]
        self.in_channels = cs[6] + cs[2] * self.block.expansion
        self.up2_gt.append(
            nn.Sequential(*self._make_layer(
                self.block, cs[6], self.num_layer[5], if_dist=if_dist))
        )
        self.up2_gt = nn.ModuleList(self.up2_gt)

        self.up3_gt = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[7],
                ks=2,
                stride=2,
                if_dist=model_cfgs.IF_DIST,
            )
        ]
        self.in_channels = cs[7] + cs[1] * self.block.expansion
        self.up3_gt.append(
            nn.Sequential(*self._make_layer(
                self.block, cs[7], self.num_layer[6], if_dist=if_dist))
        )
        self.up3_gt = nn.ModuleList(self.up3_gt)

        self.up4_gt = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[8],
                ks=2,
                stride=2,
                if_dist=model_cfgs.IF_DIST,
            )
        ]
        self.in_channels = cs[8] + cs[0]
        self.up4_gt.append(
            nn.Sequential(*self._make_layer(
                self.block, cs[8], self.num_layer[7], if_dist=if_dist))
        )
        self.up4_gt = nn.ModuleList(self.up4_gt)

        self.classifier_gt = nn.Sequential(
            nn.Linear((cs[4] + cs[6] + cs[8]) * self.block.expansion, self.num_class)
        )

        self.stem = nn.Sequential(
            spnn.Conv3d(
                self.in_feature_dim, cs[0],
                kernel_size=3,
                stride=1,
            ),
            SyncBatchNorm(cs[0]) if model_cfgs.IF_DIST else BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(
                cs[0], cs[0],
                kernel_size=3,
                stride=1,
            ),
            SyncBatchNorm(cs[0]) if model_cfgs.IF_DIST else BatchNorm(cs[0]),
            spnn.ReLU(True),
        )

        self.in_channels = cs[0]
        if_dist = model_cfgs.IF_DIST

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=model_cfgs.IF_DIST,
            ),
            *self._make_layer(
                self.block, cs[1], self.num_layer[0], if_dist=if_dist),
        )
        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=model_cfgs.IF_DIST,
            ),
            *self._make_layer(
                self.block, cs[2], self.num_layer[1], if_dist=if_dist),
        )
        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=model_cfgs.IF_DIST,
            ),
            *self._make_layer(
                self.block, cs[3], self.num_layer[2], if_dist=if_dist),
        )
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=model_cfgs.IF_DIST,
            ),
            *self._make_layer(
                self.block, cs[4], self.num_layer[3], if_dist=if_dist),
        )

        self.up1 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[5],
                ks=2,
                stride=2,
                if_dist=model_cfgs.IF_DIST,
            )
        ]
        self.in_channels = cs[5] + cs[3] * self.block.expansion
        self.up1.append(
            nn.Sequential(*self._make_layer(
                self.block, cs[5], self.num_layer[4], if_dist=if_dist))
        )
        self.up1 = nn.ModuleList(self.up1)

        self.up2 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[6],
                ks=2,
                stride=2,
                if_dist=model_cfgs.IF_DIST,
            )
        ]
        self.in_channels = cs[6] + cs[2] * self.block.expansion
        self.up2.append(
            nn.Sequential(*self._make_layer(
                self.block, cs[6], self.num_layer[5], if_dist=if_dist))
        )
        self.up2 = nn.ModuleList(self.up2)

        self.up3 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[7],
                ks=2,
                stride=2,
                if_dist=model_cfgs.IF_DIST,
            )
        ]
        self.in_channels = cs[7] + cs[1] * self.block.expansion
        self.up3.append(
            nn.Sequential(*self._make_layer(
                self.block, cs[7], self.num_layer[6], if_dist=if_dist))
        )
        self.up3 = nn.ModuleList(self.up3)

        self.up4 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[8],
                ks=2,
                stride=2,
                if_dist=model_cfgs.IF_DIST,
            )
        ]
        self.in_channels = cs[8] + cs[0]
        self.up4.append(
            nn.Sequential(*self._make_layer(
                self.block, cs[8], self.num_layer[7], if_dist=if_dist))
        )
        self.up4 = nn.ModuleList(self.up4)

        self.classifier = nn.Sequential(
            nn.Linear((cs[4] + cs[6] + cs[8]) * self.block.expansion, self.num_class)
        )

        self.weight_initialization()

        dropout_p_gt = model_cfgs.get('DROPOUT_P', 0.3)
        self.dropout_gt = nn.Dropout(dropout_p_gt, True)

        dropout_p = model_cfgs.get('DROPOUT_P', 0.3)
        self.dropout = nn.Dropout(dropout_p, True)

        label_smoothing = model_cfgs.get('LABEL_SMOOTHING', 0.0)

        # loss
        default_loss_config = {
            'LOSS_TYPES': ['CELoss', 'LovLoss'],
            'LOSS_WEIGHTS': [1.0, 1.0],
            'KNN': 10,
        }
        loss_config = self.model_cfgs.get('LOSS_CONFIG', default_loss_config)

        loss_types = loss_config.get('LOSS_TYPES', default_loss_config['LOSS_TYPES'])
        loss_weights = loss_config.get('LOSS_WEIGHTS', default_loss_config['LOSS_WEIGHTS'])
        assert len(loss_types) == len(loss_weights)
        k_nearest_neighbors = loss_config.get('KNN', default_loss_config['KNN'])

        self.criterion_losses = Losses(
            loss_types=loss_types,
            loss_weights=loss_weights,
            ignore_index=model_cfgs.IGNORE_LABEL,
            knn=k_nearest_neighbors,
            label_smoothing=label_smoothing,
        )

    def _make_layer(self, block, out_channels, num_block, stride=1, if_dist=False):
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride=stride, if_dist=if_dist)
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(
                block(self.in_channels, out_channels, if_dist=if_dist)
            )
        return layers
    
    def forward(self, batch_dict, return_logit=False, return_tta=False):
        with torch.no_grad():
            x_ms_gt = batch_dict['lidar_ms_gt']
            x_ms_gt.F = x_ms_gt.F[:, :self.in_feature_dim]
            z_ms_gt = PointTensor(x_ms_gt.F, x_ms_gt.C.float())

            x0_ms_gt = self.stem_gt(x_ms_gt)
            z0_ms_gt = voxel_to_point(x0_ms_gt, z_ms_gt, nearest=False)

            x1_ms_gt = self.stage1_gt(x0_ms_gt)
            x2_ms_gt = self.stage2_gt(x1_ms_gt)
            x3_ms_gt = self.stage3_gt(x2_ms_gt)
            x4_ms_gt = self.stage4_gt(x3_ms_gt)
            z1_ms_gt = voxel_to_point(x4_ms_gt, z0_ms_gt)

            x4_ms_gt.F = self.dropout_gt(x4_ms_gt.F)
            y1_ms_gt = self.up1_gt[0](x4_ms_gt)
            y1_ms_gt = torchsparse.cat([y1_ms_gt, x3_ms_gt])
            y1_ms_gt = self.up1_gt[1](y1_ms_gt)

            y2_ms_gt = self.up2_gt[0](y1_ms_gt)
            y2_ms_gt = torchsparse.cat([y2_ms_gt, x2_ms_gt])
            y2_ms_gt = self.up2_gt[1](y2_ms_gt)
            z2_ms_gt = voxel_to_point(y2_ms_gt, z1_ms_gt)

            y2_ms_gt.F = self.dropout_gt(y2_ms_gt.F)
            y3_ms_gt = self.up3_gt[0](y2_ms_gt)
            y3_ms_gt = torchsparse.cat([y3_ms_gt, x1_ms_gt])
            y3_ms_gt = self.up3_gt[1](y3_ms_gt)

            y4_ms_gt = self.up4_gt[0](y3_ms_gt)
            y4_ms_gt = torchsparse.cat([y4_ms_gt, x0_ms_gt])
            y4_ms_gt = self.up4_gt[1](y4_ms_gt)
            z3_ms_gt = voxel_to_point(y4_ms_gt, z2_ms_gt)

            voxel_feat_ms_gt = torch.cat([z1_ms_gt.F, z2_ms_gt.F, z3_ms_gt.F], dim=1)
            out_ms_gt = self.classifier_gt(voxel_feat_ms_gt)
            
        x_ms = batch_dict['lidar_ms']
        x_ms.F = x_ms.F[:, :self.in_feature_dim]
        z_ms = PointTensor(x_ms.F, x_ms.C.float())

        x0_ms = self.stem(x_ms)
        z0_ms = voxel_to_point(x0_ms, z_ms, nearest=False)

        x1_ms = self.stage1(x0_ms)
        x2_ms = self.stage2(x1_ms)
        x3_ms = self.stage3(x2_ms)
        x4_ms = self.stage4(x3_ms)
        z1_ms = voxel_to_point(x4_ms, z0_ms)

        x4_ms.F = self.dropout(x4_ms.F)
        y1_ms = self.up1[0](x4_ms)
        y1_ms = torchsparse.cat([y1_ms, x3_ms])
        y1_ms = self.up1[1](y1_ms)

        y2_ms = self.up2[0](y1_ms)
        y2_ms = torchsparse.cat([y2_ms, x2_ms])
        y2_ms = self.up2[1](y2_ms)
        z2_ms = voxel_to_point(y2_ms, z1_ms)

        y2_ms.F = self.dropout(y2_ms.F)
        y3_ms = self.up3[0](y2_ms)
        y3_ms = torchsparse.cat([y3_ms, x1_ms])
        y3_ms = self.up3[1](y3_ms)

        y4_ms = self.up4[0](y3_ms)
        y4_ms = torchsparse.cat([y4_ms, x0_ms])
        y4_ms = self.up4[1](y4_ms)
        z3_ms = voxel_to_point(y4_ms, z2_ms)

        voxel_feat_ms = torch.cat([z1_ms.F, z2_ms.F, z3_ms.F], dim=1)
        out_ms = self.classifier(voxel_feat_ms)

        if self.training:
            target_ms = batch_dict['targets_ms'].F.long().cuda(non_blocking=True)

            coords_xyz_ms = batch_dict['lidar_ms'].C[:, :3].float()
            offset_ms = batch_dict['offset_ms']
            loss_seg = self.criterion_losses(out_ms, target_ms, xyz=coords_xyz_ms, offset=offset_ms)

            s2d_query_1x = torchsparse.nn.functional.sphashquery(
                torchsparse.nn.functional.sphash(x_ms.C.to(x_ms.C).int()),
                torchsparse.nn.functional.sphash(x_ms_gt.C.to(x_ms_gt.C).int()))

            loss_feat_kd = 0
            batch_size = x_ms.C[:, -1].max() + 1
            for batch_idx in range(batch_size):
                batch_mask = x_ms.C[:, -1] == batch_idx
                batch_mask_gt = x_ms_gt.C[:, -1] == batch_idx

                if self.sampling_type in ['random'] :
                    distill_mask = (s2d_query_1x>=0) & batch_mask

                if self.sampling_type in ['random']:
                    distill_mask_sample = copy.deepcopy(distill_mask)
                    if distill_mask.sum() > self.max_voxel:
                        distill_inds = distill_mask.nonzero().reshape(-1)
                        distill_mask_sample[distill_inds[torch.randperm(len(distill_inds))][self.max_voxel:]] = False
                    voxel_feat_s = [voxel_feat_ms[distill_mask_sample]]
                    voxel_feat_t = [voxel_feat_ms_gt[s2d_query_1x[distill_mask_sample]].detach()]
                    loss_feat_kd += sum(self.criterion_feat_kd(voxel_feat_s, voxel_feat_t)) * self.feat_kd_weight / batch_size

            loss = loss_seg + loss_feat_kd

            ret_dict = {'loss': loss}
            disp_dict = {'loss': loss.item(),'loss_seg': loss_seg.item(),'loss_feat_kd': loss_feat_kd.item()}
            tb_dict = {'loss': loss.item(),'loss_seg': loss_seg.item(),'loss_feat_kd': loss_feat_kd.item()}
            return ret_dict, tb_dict, disp_dict
        else:
            invs_ms = batch_dict['inverse_map_ms']
            all_labels = batch_dict['targets_mapped']
            point_mask = batch_dict['point_mask']
            num_points_ms = batch_dict['num_points_ms']
            point_predict = []
            point_labels = []
            point_predict_logits = []
            pointer = 0
            for idx in range(invs_ms.C[:, -1].max() + 1):
                cur_scene_pts_ms = (x_ms.C[:, -1] == idx).cpu().numpy()
                cur_inv_ms = invs_ms.F[invs_ms.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                if return_logit or return_tta:
                    outputs_mapped = out_ms[cur_scene_pts_ms][cur_inv_ms].softmax(1)[point_mask[pointer: pointer+num_points_ms[idx]]]
                else:
                    outputs_mapped = out_ms[cur_scene_pts_ms][cur_inv_ms].argmax(1)[point_mask[pointer: pointer+num_points_ms[idx]]]
                    outputs_mapped_logits = out_ms[cur_scene_pts_ms][cur_inv_ms][point_mask[pointer: pointer+num_points_ms[idx]]]
                targets_mapped = all_labels.F[cur_label]

                point_predict.append(outputs_mapped[:batch_dict['num_points'][idx]].cpu().numpy())
                point_labels.append(targets_mapped[:batch_dict['num_points'][idx]].cpu().numpy())
                point_predict_logits.append(outputs_mapped_logits[:batch_dict['num_points'][idx]].cpu().numpy())
                pointer += num_points_ms[idx]

            return {'point_predict': point_predict, 'point_labels': point_labels, 'name': batch_dict['name'],'point_predict_logits': point_predict_logits}

    def forward_ensemble(self, batch_dict):
        return self.forward(batch_dict, ensemble=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        model_state_disk = torch.load(filename, map_location=loc_type)
        if 'model_state' in model_state_disk:
            model_state_disk = model_state_disk['model_state']
        
        model_state_disk_new = {}
        for k in model_state_disk.keys():
            model_state_disk_new[k] = model_state_disk[k]
            if 'stem' in k:
                k_new = k.replace('stem', 'stem_gt')
            if 'stage1' in k:
                k_new = k.replace('stage1', 'stage1_gt')
            if 'stage2' in k:
                k_new = k.replace('stage2', 'stage2_gt')
            if 'stage3' in k:
                k_new = k.replace('stage3', 'stage3_gt')
            if 'stage4' in k:
                k_new = k.replace('stage4', 'stage4_gt')
            if 'up1' in k:
                k_new = k.replace('up1', 'up1_gt')
            if 'up2' in k:
                k_new = k.replace('up2', 'up2_gt')
            if 'up3' in k:
                k_new = k.replace('up3', 'up3_gt')
            if 'up4' in k:
                k_new = k.replace('up4', 'up4_gt')
            if 'dropout' in k:
                k_new = k.replace('dropout', 'dropout_gt')
            if 'classifier' in k:
                k_new = k.replace('classifier', 'classifier_gt')
            model_state_disk_new[k_new] = model_state_disk[k]

        msg = self.load_params(model_state_disk_new)
        logger.info(f"==> Done {msg}")

    def fix_part_param(self, keywards='gt'):
        for k,v in self.named_parameters():
            if keywards in k:
                v.requires_grad=False