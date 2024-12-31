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
from .utils import initial_voxelize, voxel_to_point, voxel_to_point_fov
from .unet2d import UNet2D
from .unet3d import UNet3D
from pcseg.loss import Losses
import numpy as np
import cv2
import pickle
import copy

__all__ = ['MinkUNetMsMmNus']

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


class MinkUNetMsMmNus(BaseSegmentor):    
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

        self.input_feat = model_cfgs.get('INPUT_FEAT')
        self.input_feat_lidar = model_cfgs.get('INPUT_FEAT_LIDAR')
        self.image_channel_in = 0
        if 'rgb' in self.input_feat:
            self.image_channel_in += 3
        if 'depth' in self.input_feat:
            self.image_channel_in += 1
        if 'lidar' in self.input_feat:
            self.image_channel_in += 4
        self.image_backbone_type = model_cfgs.get('IMAGE_BACKBONE_TYPE')
        if self.image_backbone_type == 'UNet2D':
            self.image_backbone = UNet2D(self.image_channel_in, self.num_class)
            self.image_channel_out = 96 + 128

        self.lidar_channel_in = 0
        if 'lidar' in self.input_feat_lidar:
            self.lidar_channel_in += self.in_feature_dim-1
        if 'image' in self.input_feat_lidar:
            self.lidar_channel_in += self.image_channel_out
        if 'logit' in self.input_feat_lidar:
            self.lidar_channel_in += self.num_class
        self.lidar_backbone_type = model_cfgs.get('LIDAR_BACKBONE_TYPE')
        if self.lidar_backbone_type == 'UNet3D':
            self.lidar_backbone = UNet3D(self.lidar_channel_in, self.num_class)

        self.lidar_weight, self.fusion_weight, self.image_weight_s, self.image_weight_d, self.image_lidar_weight = model_cfgs.get('LOSS_WEIGHT')
        self.fusion_type = model_cfgs.get('FUSION_TYPE')
        self.ensemble_type = model_cfgs.get('ENSEMBLE_TYPE')

        self.fusion_channel = (cs[4] + cs[6] + cs[8]) * self.block.expansion
        if self.fusion_type == 'cat':
            self.fusion_channel += ((cs[4] + cs[6] + cs[8]) * self.block.expansion)
            
        self.classifier_fusion = nn.Sequential(
            nn.Linear(self.fusion_channel, (cs[4] + cs[6] + cs[8]) * self.block.expansion),
            nn.BatchNorm1d((cs[4] + cs[6] + cs[8]) * self.block.expansion),
            nn.ReLU(inplace=True),
            nn.Linear((cs[4] + cs[6] + cs[8]) * self.block.expansion, self.num_class)
        )
        
        self.weight_initialization()

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

    def _make_layer_fov(self, block, out_channels, num_block, stride=1, if_dist=False):
        layers = []
        layers.append(
            block(self.in_channels_fov, out_channels, stride=stride, if_dist=if_dist)
        )
        self.in_channels_fov = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(
                block(self.in_channels_fov, out_channels, if_dist=if_dist)
            )
        return layers
    
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict, return_logit=False, return_tta=False):
        image_input = []
        if 'rgb' in self.input_feat:
            image_input.append(batch_dict['image_ms'])
        if 'depth' in self.input_feat:
            image_input.append(batch_dict['depth_map_ms'])
        if 'lidar' in self.input_feat:
            image_input.append(batch_dict['lidar_map_ms'])
        batch_dict['image_input'] = torch.cat(image_input, dim=1)
        batch_dict = self.image_backbone(batch_dict)
        image_logits = batch_dict['image_logits'].permute(0, 2, 3, 1).reshape(-1, self.num_class)
        image_targets = batch_dict['semantic_map_ms'].permute(0, 2, 3, 1).reshape(-1)
        image_logits_fov = batch_dict['image_logits_fov']
        image_targets_fov = batch_dict['targets_fov_ms'].F.reshape(-1)
        targets_fov_ms = batch_dict['targets_fov_ms'].F.reshape(-1)
        image_features_fov = batch_dict['image_features_fov']
        x_fov_ms = batch_dict['lidar_fov_ms']
        x_fov_ms_uv = batch_dict['lidar_fov_ms'].F[:, -2:]
        lidar_input = []
        if 'lidar' in self.input_feat_lidar:
            lidar_input.append(x_fov_ms.F[:, :self.in_feature_dim-1])
        if 'image' in self.input_feat_lidar:
            lidar_input.append(image_features_fov)
        if 'logit' in self.input_feat_lidar:
            lidar_input.append(image_logits_fov)
        x_fov_ms.F = torch.cat(lidar_input, dim=-1)
        image_lidar_logits_fov, x4_fov_ms, y2_fov_ms, y4_fov_ms = self.lidar_backbone(batch_dict)

        x_ms = batch_dict['lidar_ms']
        x_ms.F = x_ms.F[:, :self.in_feature_dim]
        # x: SparseTensor z: PointTensor
        z_ms = PointTensor(x_ms.F, x_ms.C.float())

        x0_ms = self.stem(x_ms)
        z0_ms = voxel_to_point(x0_ms, z_ms, nearest=False)

        x1_ms = self.stage1(x0_ms)
        x2_ms = self.stage2(x1_ms)
        x3_ms = self.stage3(x2_ms)
        x4_ms = self.stage4(x3_ms)
        z1_ms = voxel_to_point(x4_ms, z0_ms)
        z1_fov_ms = voxel_to_point_fov(x4_fov_ms, z0_ms)

        x4_ms.F = self.dropout(x4_ms.F)
        y1_ms = self.up1[0](x4_ms)
        y1_ms = torchsparse.cat([y1_ms, x3_ms])
        y1_ms = self.up1[1](y1_ms)

        y2_ms = self.up2[0](y1_ms)
        y2_ms = torchsparse.cat([y2_ms, x2_ms])
        y2_ms = self.up2[1](y2_ms)
        z2_ms = voxel_to_point(y2_ms, z1_ms)
        z2_fov_ms = voxel_to_point_fov(y2_fov_ms, z1_ms)

        y2_ms.F = self.dropout(y2_ms.F)
        y3_ms = self.up3[0](y2_ms)
        y3_ms = torchsparse.cat([y3_ms, x1_ms])
        y3_ms = self.up3[1](y3_ms)

        y4_ms = self.up4[0](y3_ms)
        y4_ms = torchsparse.cat([y4_ms, x0_ms])
        y4_ms = self.up4[1](y4_ms)
        z3_ms = voxel_to_point(y4_ms, z2_ms)
        z3_fov_ms = voxel_to_point_fov(y4_fov_ms, z2_ms)
        if (z3_ms.F.sum(-1)==0).sum()>0:
            print('empty feature of minkunet: %d'%(z3_ms.F.sum(-1)==0).sum())

        lidar_features = torch.cat([z1_ms.F, z2_ms.F, z3_ms.F], dim=1)
        out_ms = self.classifier(lidar_features)

        if self.fusion_type == 'cat':
            overlap_mask = z1_fov_ms.F.sum(-1)!=0
            fusion_features= torch.cat([z1_ms.F, z2_ms.F, z3_ms.F, z1_fov_ms.F, z2_fov_ms.F, z3_fov_ms.F], dim=1)

        out_ms_fusion = self.classifier_fusion(fusion_features[overlap_mask])

        if self.training:
            target_ms = batch_dict['targets_ms'].F.long().cuda(non_blocking=True)
            coords_xyz_ms = batch_dict['lidar_ms'].C[:, :3].float()
            offset_ms = batch_dict['offset_ms']

            loss_lidar = self.criterion_losses(out_ms, target_ms, xyz=coords_xyz_ms, offset=offset_ms) * self.lidar_weight
            loss_fusion = self.criterion_losses(out_ms_fusion, target_ms[overlap_mask]) * self.fusion_weight
            loss_image_s = self.criterion_losses(image_logits_fov, image_targets_fov.to(target_ms.dtype)) * self.image_weight_s
            loss_image_d = self.criterion_losses(image_logits, image_targets.to(target_ms.dtype)) * self.image_weight_d
            loss_image_lidar = self.criterion_losses(image_lidar_logits_fov, targets_fov_ms.to(target_ms.dtype)) * self.image_lidar_weight

            loss = loss_lidar + loss_fusion + loss_image_s + loss_image_d + loss_image_lidar
            ret_dict = {'loss': loss}
            disp_dict = {'loss': loss.item(),'loss_lidar': loss_lidar.item(),'loss_fusion': loss_fusion.item(),\
                    'loss_image_s': loss_image_s.item(),'loss_image_d': loss_image_d.item(),'loss_image_lidar': loss_image_lidar.item()}
            tb_dict = {'loss': loss.item(),'loss_lidar': loss_lidar.item(),'loss_fusion': loss_fusion.item(),\
                    'loss_image_s': loss_image_s.item(),'loss_image_d': loss_image_d.item(),'loss_image_lidar': loss_image_lidar.item()}

            return ret_dict, tb_dict, disp_dict
        else:
            if self.ensemble_type == 'replace':
                out_ms[overlap_mask] = out_ms_fusion

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

    def fix_part_param(self):
        for k,v in self.named_parameters():
            if ('image_backbone' not in k) and ('classifier_fusion' not in k) and ('lidar_backbone' not in k):
                v.requires_grad=False
