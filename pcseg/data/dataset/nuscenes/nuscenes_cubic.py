'''
This file is modified from https://github.com/mit-han-lab/spvnas
'''


import numpy as np
import torch
from torch.utils import data
from .nuscenes import NuscenesDataset
from torchsparse.utils.quantize import sparse_quantize
from itertools import accumulate
from tools.utils.common.seg_utils import aug_points
from collections import defaultdict


def voxelize_with_label(point_coords, point_labels, num_classes):
    """
        point_coords [N, 3](x,y,z): point coords on the voxel space
        point_labels [N]
        voxel_coords [M, 3](x,y,z): voxel coords
    """
    voxel_coords, inds, inverse_map = sparse_quantize(point_coords, return_index=True, return_inverse=True)
    voxel_label_counter = np.zeros([voxel_coords.shape[0], num_classes])
    for ind in range(len(inverse_map)):
        if point_labels[ind] != 67:
            voxel_label_counter[inverse_map[ind]][point_labels[ind]] += 1
    voxel_labels = np.argmax(voxel_label_counter, axis=1)

    return voxel_coords, voxel_labels, inds, inverse_map


class NuscCubicDataset(data.Dataset):
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

        self.point_cloud_dataset = NuscenesDataset(
            data_cfgs=data_cfgs,
            training=training,
            class_names=self.class_names,
            root_path=self.root_path,
            logger=logger,
        )

        self.voxel_size = data_cfgs.VOXEL_SIZE
        self.num_points = data_cfgs.NUM_POINTS

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
        point = pc_data['xyzret'][:,:4].astype(np.float32)

        num_points_current_frame = point.shape[0]
        ret = {}
        if self.training:
            point[:, 0:3] = aug_points(
                xyz=point[:, :3],
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
            self.if_scale = True
            self.scale_aug_range = [0.95, 1.05]
            self.if_jitter = False
            self.if_rotate = True
            point[:, 0:3] = aug_points(
                xyz=point[:, :3],
                if_flip=self.if_flip,
                if_scale=self.if_scale,
                scale_axis=self.scale_axis,
                scale_range=self.scale_range,
                if_jitter=self.if_jitter,
                if_rotate=self.if_rotate,
                if_tta=True,
                num_vote=voting_idx,
        )
        xyz_pol = point[:,:3]
        point_coord = np.round(point[:, :3] / 0.05).astype(np.int32)
        point_coord -= point_coord.min(0, keepdims=1)

        voxel_coord, voxel_label, inds, inverse_map = voxelize_with_label(
            point_coord, point_label, len(self.class_names))
        voxel_feature = np.concatenate([xyz_pol[inds], point[inds][:, 3:]], axis=1)
        point_feature = np.concatenate([xyz_pol,  point[:, 3:]], axis=1)
        ret.update(
            {
                'name': pc_data['path'],
                'point_feature': point_feature.astype(np.float32),
                'point_coord': point_coord.astype(np.float32),
                'point_label': point_label.astype(np.int),
                'voxel_feature': voxel_feature.astype(np.float32),
                'voxel_coord': voxel_coord.astype(np.int),
                'voxel_label': voxel_label.astype(np.int),
                'inverse_map': inverse_map.astype(np.int),
                'num_points': np.array([num_points_current_frame]), # for multi frames
            })
        return ret

    @staticmethod
    def collate_batch(batch_list):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        point_coord = []
        voxel_coord = []
        for i_batch in range(batch_size):
            point_coord.append(
                np.pad(data_dict['point_coord'][i_batch], ((0, 0), (0, 1)), mode='constant', constant_values=i_batch))
            voxel_coord.append(
                np.pad(data_dict['voxel_coord'][i_batch], ((0, 0), (0, 1)), mode='constant', constant_values=i_batch))

        ret['point_coord'] = torch.from_numpy(np.concatenate(point_coord)).type(torch.LongTensor)
        ret['voxel_coord'] = torch.from_numpy(np.concatenate(voxel_coord)).type(torch.LongTensor)

        ret['point_feature'] = torch.from_numpy(np.concatenate(data_dict['point_feature'])).type(torch.FloatTensor)
        ret['point_label'] = torch.from_numpy(np.concatenate(data_dict['point_label'])).type(torch.LongTensor)
        ret['voxel_feature'] = torch.from_numpy(np.concatenate(data_dict['voxel_feature'])).type(torch.FloatTensor)
        ret['voxel_label'] = torch.from_numpy(np.concatenate(data_dict['voxel_label'])).type(torch.LongTensor)
        ret['inverse_map'] = torch.from_numpy(np.concatenate(data_dict['inverse_map'])).type(torch.LongTensor)
        ret['num_points']= torch.from_numpy(np.concatenate(data_dict['num_points'])).type(torch.LongTensor)
        offset = [sample['voxel_coord'].shape[0] for sample in batch_list] 
        ret['offset'] = torch.tensor(list(accumulate(offset))).int()
        ret['name'] = data_dict['name']

        for k, v in data_dict.items():
            if k.startswith('flag'):
                ret[k] = data_dict[k]
            elif k.startswith('augmented_point_coord'):
                temp = []
                for i_batch in range(batch_size):
                    temp.append(
                        np.pad(data_dict[k][i_batch], ((0, 0), (0, 1)), mode='constant', constant_values=i_batch))
                ret[k] = torch.from_numpy(np.concatenate(temp)).type(torch.LongTensor)
            elif k.startswith('augmented_point_feature'):
                ret[k] = torch.from_numpy(np.concatenate(data_dict[k])).type(torch.FloatTensor)
            elif k.startswith('augmented_point_label') or k.startswith('augmented_inverse_map'):
                ret[k] = torch.from_numpy(np.concatenate(data_dict[k])).type(torch.LongTensor)

        return ret
    
    @staticmethod
    def collate_batch_tta(batch_list):
        batch_list = batch_list[0]
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        point_coord = []
        voxel_coord = []
        for i_batch in range(batch_size):
            point_coord.append(
                np.pad(data_dict['point_coord'][i_batch], ((0, 0), (0, 1)), mode='constant', constant_values=i_batch))
            voxel_coord.append(
                np.pad(data_dict['voxel_coord'][i_batch], ((0, 0), (0, 1)), mode='constant', constant_values=i_batch))

        ret['point_coord'] = torch.from_numpy(np.concatenate(point_coord)).type(torch.LongTensor)
        ret['voxel_coord'] = torch.from_numpy(np.concatenate(voxel_coord)).type(torch.LongTensor)

        ret['point_feature'] = torch.from_numpy(np.concatenate(data_dict['point_feature'])).type(torch.FloatTensor)
        ret['point_label'] = torch.from_numpy(np.concatenate(data_dict['point_label'])).type(torch.LongTensor)
        ret['voxel_feature'] = torch.from_numpy(np.concatenate(data_dict['voxel_feature'])).type(torch.FloatTensor)
        ret['voxel_label'] = torch.from_numpy(np.concatenate(data_dict['voxel_label'])).type(torch.LongTensor)
        ret['inverse_map'] = torch.from_numpy(np.concatenate(data_dict['inverse_map'])).type(torch.LongTensor)
        ret['num_points']= torch.from_numpy(np.concatenate(data_dict['num_points'])).type(torch.LongTensor)
        offset = [sample['voxel_coord'].shape[0] for sample in batch_list] 
        ret['offset'] = torch.tensor(list(accumulate(offset))).int()
        ret['name'] = data_dict['name']

        for k, v in data_dict.items():
            if k.startswith('flag'):
                ret[k] = data_dict[k]
            elif k.startswith('augmented_point_coord'):
                temp = []
                for i_batch in range(batch_size):
                    temp.append(
                        np.pad(data_dict[k][i_batch], ((0, 0), (0, 1)), mode='constant', constant_values=i_batch))
                ret[k] = torch.from_numpy(np.concatenate(temp)).type(torch.LongTensor)
            elif k.startswith('augmented_point_feature'):
                ret[k] = torch.from_numpy(np.concatenate(data_dict[k])).type(torch.FloatTensor)
            elif k.startswith('augmented_point_label') or k.startswith('augmented_inverse_map'):
                ret[k] = torch.from_numpy(np.concatenate(data_dict[k])).type(torch.LongTensor)

        return ret

    #     pc_ = np.round(point[:, :3] / self.voxel_size).astype(np.int32)
    #     pc_ -= pc_.min(0, keepdims=1)
    #     feat_ = point
    #     _, inds, inverse_map = sparse_quantize(
    #         pc_,
    #         return_index=True,
    #         return_inverse=True,
    #     )
    #     if self.training and len(inds) > self.num_points:  # NOTE: num_points must always bigger than self.num_points
    #         raise RuntimeError('droping point')
    #         inds = np.random.choice(inds, self.num_points, replace=False)

    #     pc = pc_[inds]
    #     feat = feat_[inds]
    #     labels = point_label[inds]
        
    #     lidar = SparseTensor(feat, pc)
    #     labels = SparseTensor(labels, pc)
    #     labels_ = SparseTensor(point_label, pc_)
    #     inverse_map = SparseTensor(inverse_map, pc_)
    #     ret = {
    #         'name': pc_data['path'],
    #         'lidar': lidar,
    #         'targets': labels,
    #         'targets_mapped': labels_,
    #         'inverse_map': inverse_map,
    #         'num_points': np.array([num_points_current_frame]), # for multi frames
    #     }

    #     return ret

    # @staticmethod
    # def collate_batch(inputs):
    #     offset = [sample['lidar'].C.shape[0] for sample in inputs] # for point transformer
    #     offsets = {}
    #     ret = sparse_collate_fn(inputs)
    #     ret.update(dict(
    #         offset=torch.tensor(list(accumulate(offset))).int()
    #     ))
    #     return ret
    
    # @staticmethod
    # def collate_batch_tta(inputs):
    #     inputs = inputs[0]
    #     offset = [sample['lidar'].C.shape[0] for sample in inputs] # for point transformer
    #     offsets = {}

    #     ret = sparse_collate_fn(inputs)
    #     ret.update(dict(
    #         offset=torch.tensor(list(accumulate(offset))).int()
    #     ))
    #     return ret

