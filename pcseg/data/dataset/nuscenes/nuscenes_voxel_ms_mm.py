'''
This file is modified from https://github.com/mit-han-lab/spvnas
'''


import numpy as np
import torch
from torch.utils import data
from .nuscenes_ms_mm import NuscenesMsMmDataset
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from itertools import accumulate
from tools.utils.common.seg_utils import aug_points_rgb_ms
import pickle
import copy

class NuscVoxelMsMmDataset(data.Dataset):
    def __init__(
        self,
        data_cfgs=None,
        training=True,
        root_path=None,
        logger=None,
    ):
        super().__init__()
        self.data_cfgs = data_cfgs
        self.training = training
        self.class_names = [
            "noise",  # ignored
            "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone",
            "trailer", "truck", "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation"
        ]
        
        self.root_path = root_path if root_path is not None else self.data_cfgs.DATA_PATH
        self.logger = logger

        if self.data_cfgs.DATASET == 'nuscenes_ms_mm':
            self.point_cloud_dataset = NuscenesMsMmDataset(
                data_cfgs=data_cfgs,
                training=training,
                class_names=self.class_names,
                root_path=self.root_path,
                logger=logger,
            )

        self.voxel_size = data_cfgs.VOXEL_SIZE
        self.num_points = data_cfgs.NUM_POINTS
        self.in_feature_dim = data_cfgs.get('IN_FEATURE_DIM', 4)

        self.if_flip = data_cfgs.get('FLIP_AUG', True)
        self.if_scale = data_cfgs.get('SCALE_AUG', True)
        self.scale_axis = data_cfgs.get('SCALE_AUG_AXIS', 'xyz')
        self.scale_range = data_cfgs.get('SCALE_AUG_RANGE', [0.9, 1.1])
        self.if_jitter = data_cfgs.get('TRANSFORM_AUG', True)
        self.if_rotate = data_cfgs.get('ROTATE_AUG', True)

        self.if_tta = self.data_cfgs.get('TTA', False)
        self.votes_min = self.data_cfgs.get('VOTES_MIN', 0)
        self.votes_max = self.data_cfgs.get('VOTES_MAX', 10)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)


    def __getitem__(self, index):
        if self.if_tta:
            data_total = []
            for idx in range(self.votes_min, self.votes_max):
                data_single = self.get_single_sample(index, idx)
                data_total.append(data_single)
            return data_total
        else:
            data = self.get_single_sample(index)
            return data

    def get_single_sample(self, index, voting_idx=0):
        'Generates one sample of data'
        pc_data = self.point_cloud_dataset[index]
        point_label = pc_data['labels'].reshape(-1)
        point = pc_data['xyzret'][:,:self.in_feature_dim].astype(np.float32)
        point_label_ms = pc_data['labels_ms'].reshape(-1)
        point_ms = pc_data['xyzret_ms'][:, :self.in_feature_dim].astype(np.float32) # pc_data['xyzret_ms'][:, :5]
        point_label_fov_ms = pc_data['labels_fov_ms'].reshape(-1)
        point_fov_ms = pc_data['xyzret_fov_ms']

        num_points_current_frame = point.shape[0]
        num_points_current_frame_ms = point_ms.shape[0]

        ret = {}
        if self.training:
            point[:, 0:3], point_ms[:, 0:3], point_fov_ms[:, 0:3] = aug_points_rgb_ms(
                xyz=point[:, :3],
                xyz_ms=point_ms[:, :3],
                xyz_rgb_ms=point_fov_ms[:, :3],
                if_flip=self.if_flip,
                if_scale=self.if_scale,
                scale_axis=self.scale_axis,
                scale_range=self.scale_range,
                if_jitter=self.if_jitter,
                if_rotate=self.if_rotate,
                if_tta=self.if_tta,
            )

        elif self.if_tta:
            self.if_flip = False
            # self.if_flip = True
            self.if_scale = True
            self.scale_aug_range = [0.95, 1.05]
            self.if_jitter = False
            self.if_rotate = True
            point[:, 0:3], point_ms[:, 0:3], point_fov_ms[:, 0:3] = aug_points_rgb_ms(
                xyz=point[:, :3],
                xyz_ms=point_ms[:, :3],
                xyz_rgb_ms=point_fov_ms[:, :3],
                if_flip=self.if_flip,
                if_scale=self.if_scale,
                scale_axis=self.scale_axis,
                scale_range=self.scale_range,
                if_jitter=self.if_jitter,
                if_rotate=self.if_rotate,
                if_tta=True,
                num_vote=voting_idx,
        )

        clamp_mask = (point_ms[:, 0] >= point[:, 0].min()) & (point_ms[:, 1] >= point[:, 1].min()) & (point_ms[:, 2] >= point[:, 2].min())
        point_label_ms = point_label_ms[clamp_mask]
        point_ms = point_ms[clamp_mask]
        num_points_current_frame_ms = point_ms.shape[0]
        assert (point[:, 0].min() == point_ms[:, 0].min()) and (point[:, 1].min() == point_ms[:, 1].min()) and (point[:, 2].min() == point_ms[:, 2].min())

        clamp_fov_mask = (point_fov_ms[:, 0] >= point[:, 0].min()) & (point_fov_ms[:, 1] >= point[:, 1].min()) & (point_fov_ms[:, 2] >= point[:, 2].min())
        point_label_fov_ms = point_label_fov_ms[clamp_fov_mask]
        point_fov_ms = point_fov_ms[clamp_fov_mask]

        pc_ = np.round(point[:, :3] / self.voxel_size).astype(np.int32)
        pc_ms_ = np.round(point_ms[:, :3] / self.voxel_size).astype(np.int32)
        pc_fov_ms_ = np.round(point_fov_ms[:, :3] / self.voxel_size).astype(np.int32)
        coord_shift = copy.deepcopy(pc_ms_.min(0, keepdims=1))

        pc_ -= coord_shift
        feat_ = point
        _, inds, inverse_map = sparse_quantize(
            pc_,
            return_index=True,
            return_inverse=True,
        
        )
        if self.training and len(inds) > self.num_points:  # NOTE: num_points must always bigger than self.num_points
            raise RuntimeError('droping point')
            inds = np.random.choice(inds, self.num_points, replace=False)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = point_label[inds]
        
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(point_label, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        pc_ms_ -= coord_shift
        feat_ms_ = point_ms
        _, inds_ms, inverse_map_ms = sparse_quantize(
            copy.deepcopy(pc_ms_),
            return_index=True,
            return_inverse=True,
        
        )
        if self.training and len(inds_ms) > self.num_points:  # NOTE: num_points must always bigger than self.num_points
            raise RuntimeError('droping point')
            inds_ms = np.random.choice(inds_ms, self.num_points, replace=False)

        pc_ms = pc_ms_[inds_ms]
        feat_ms = feat_ms_[inds_ms]
        labels_ms = point_label_ms[inds_ms]
        
        lidar_ms = SparseTensor(feat_ms, pc_ms)
        labels_ms = SparseTensor(labels_ms, pc_ms)
        labels_ms_ = SparseTensor(point_label_ms, pc_ms_)
        inverse_map_ms = SparseTensor(inverse_map_ms, pc_ms_)

        pc_fov_ms_ -= coord_shift
        feat_fov_ms_ = point_fov_ms
        _, inds_fov_ms, inverse_map_fov_ms = sparse_quantize(
            copy.deepcopy(pc_fov_ms_),
            return_index=True,
            return_inverse=True,
        
        )
        pc_fov_ms = pc_fov_ms_[inds_fov_ms]
        feat_fov_ms = feat_fov_ms_[inds_fov_ms]
        labels_fov_ms = point_label_fov_ms[inds_fov_ms]

        lidar_fov_ms = SparseTensor(feat_fov_ms, pc_fov_ms)
        labels_fov_ms = SparseTensor(labels_fov_ms, pc_ms)

        ret = {
            'name': pc_data['path'],
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'num_points': np.array([num_points_current_frame]), # for multi frames

            'lidar_ms': lidar_ms,
            'targets_ms': labels_ms,
            'targets_mapped_ms': labels_ms_,
            'inverse_map_ms': inverse_map_ms,
            'num_points_ms': np.array([num_points_current_frame_ms]), # for multi frames

            'lidar_fov_ms': lidar_fov_ms,
            'targets_fov_ms': labels_fov_ms,
            'image_ms': pc_data['image_ms'],
            'depth_map_ms': pc_data['depth_map_ms'],
            'lidar_map_ms': pc_data['lidar_map_ms'],
            'semantic_map_ms': pc_data['semantic_map_ms'],
        }

        return ret

    @staticmethod
    def collate_batch(inputs):
        offsets = {}

        offset_image = [sample['image_ms'].shape[0] for sample in inputs] # for point transformer

        image_ms = [torch.from_numpy(sample['image_ms']) for sample in inputs]
        [sample.pop('image_ms') for sample in inputs]

        depth_map_ms = [torch.from_numpy(sample['depth_map_ms']) for sample in inputs]
        [sample.pop('depth_map_ms') for sample in inputs]

        lidar_map_ms = [torch.from_numpy(sample['lidar_map_ms']) for sample in inputs]
        [sample.pop('lidar_map_ms') for sample in inputs]

        semantic_map_ms = [torch.from_numpy(sample['semantic_map_ms']) for sample in inputs]
        [sample.pop('semantic_map_ms') for sample in inputs]

        ret = sparse_collate_fn(inputs)

        ret.update(dict(
            offset_img=torch.tensor(list(accumulate(offset_image))).int(),
            image_ms=torch.cat(image_ms, axis=0).permute(0, 3, 1, 2).contiguous(),
            depth_map_ms=torch.cat(depth_map_ms, axis=0).permute(0, 3, 1, 2).contiguous(),
            lidar_map_ms=torch.cat(lidar_map_ms, axis=0).permute(0, 3, 1, 2).contiguous(),
            semantic_map_ms=torch.cat(semantic_map_ms, axis=0).permute(0, 3, 1, 2).contiguous(),
        ))

        offset = [sample['lidar'].C.shape[0] for sample in inputs] # for point transformer
        ret.update(dict(
            offset=torch.tensor(list(accumulate(offset))).int()
        ))

        offset_ms = [sample['lidar_ms'].C.shape[0] for sample in inputs] # for point transformer
        ret.update(dict(
            offset_ms=torch.tensor(list(accumulate(offset_ms))).int()
        ))

        batch_size = len(inputs)
        point_mask = torch.zeros(ret['num_points_ms'].sum(), dtype=torch.bool)
        cur = 0
        for i_batch in range(batch_size):
                point_mask[cur: cur+ret['num_points'][i_batch]] = True
                cur += ret['num_points_ms'][i_batch]
        ret['point_mask'] = point_mask

        return ret
    
    @staticmethod
    def collate_batch_tta(inputs):
        inputs = inputs[0]
        offsets = {}

        offset_image = [sample['image_ms'].shape[0] for sample in inputs] # for point transformer

        image_ms = [torch.from_numpy(sample['image_ms']) for sample in inputs]
        [sample.pop('image_ms') for sample in inputs]

        depth_map_ms = [torch.from_numpy(sample['depth_map_ms']) for sample in inputs]
        [sample.pop('depth_map_ms') for sample in inputs]

        lidar_map_ms = [torch.from_numpy(sample['lidar_map_ms']) for sample in inputs]
        [sample.pop('lidar_map_ms') for sample in inputs]

        semantic_map_ms = [torch.from_numpy(sample['semantic_map_ms']) for sample in inputs]
        [sample.pop('semantic_map_ms') for sample in inputs]

        ret = sparse_collate_fn(inputs)
        
        ret.update(dict(
            offset_img=torch.tensor(list(accumulate(offset_image))).int(),
            image_ms=torch.cat(image_ms, axis=0).permute(0, 3, 1, 2).contiguous(),
            depth_map_ms=torch.cat(depth_map_ms, axis=0).permute(0, 3, 1, 2).contiguous(),
            lidar_map_ms=torch.cat(lidar_map_ms, axis=0).permute(0, 3, 1, 2).contiguous(),
            semantic_map_ms=torch.cat(semantic_map_ms, axis=0).permute(0, 3, 1, 2).contiguous(),
        ))

        offset = [sample['lidar'].C.shape[0] for sample in inputs] # for point transformer
        ret.update(dict(
            offset=torch.tensor(list(accumulate(offset))).int()
        ))

        offset_ms = [sample['lidar_ms'].C.shape[0] for sample in inputs] # for point transformer
        ret.update(dict(
            offset_ms=torch.tensor(list(accumulate(offset_ms))).int()
        ))

        batch_size = len(inputs)
        point_mask = torch.zeros(ret['num_points_ms'].sum(), dtype=torch.bool)
        cur = 0
        for i_batch in range(batch_size):
                point_mask[cur: cur+ret['num_points'][i_batch]] = True
                cur += ret['num_points_ms'][i_batch]
        ret['point_mask'] = point_mask

        return ret

