import os
import numpy as np
from torch.utils import data
from .LaserMix_nuscenes import lasermix_aug, lasermix_aug_
from .PolarMix_nuscenes import polarmix
from pyquaternion import Quaternion
from typing import List, Tuple, Union
from itertools import repeat
import random
import pickle
import yaml
import pdb

# used for polarmix
instance_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]

class NuscenesMsDataset(data.Dataset):
    def __init__(
        self,
        data_cfgs=None,
        training: bool = True,
        class_names: list = None,
        root_path: str = None,
        logger = None,
    ):
        super().__init__()
        self.data_cfgs = data_cfgs
        self.root_path = root_path
        self.training = training
        self.logger = logger
        self.class_names = class_names
        self.tta = data_cfgs.get('TTA', False)
        self.seq = data_cfgs.get('SEQ', -1)
        self.train_val = data_cfgs.get('TRAINVAL', False)
        self.augment = data_cfgs.AUGMENT

        from pcseg.data.dataset.ceph import PetrelBackend
        self.petrel_client = PetrelBackend()
        self.data_path_ceph = data_cfgs.get('DATA_PATH_CEPH', None) # "cluster3:s3://wxp-DataSets/nuscenes/"

        if self.training:
            self.split = 'train'
        else:
            self.split = 'val'
        if self.tta and self.seq == -1:
            self.split = 'test'

        from nuscenes import NuScenes
        if self.split == 'test' and self.seq == -1:
            self.nusc = NuScenes(version='v1.0-test', dataroot=root_path, verbose=True)
        else:
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=root_path, verbose=True)

        if (self.split == 'train') and self.train_val:
            with open(os.path.join(self.root_path, data_cfgs.INFO_PATH['train']), 'rb') as f:
                data_train = pickle.load(f)
            with open(os.path.join(self.root_path, data_cfgs.INFO_PATH['val']), 'rb') as f:
                data_val = pickle.load(f)
            self.nusc_infos = data_train['infos'] + data_val['infos']
        else:
            with open(os.path.join(self.root_path, data_cfgs.INFO_PATH[self.split]), 'rb') as f:
                data = pickle.load(f)
            self.nusc_infos = data['infos']
        print(f'The total sample is {len(self.nusc_infos)}')

        with open('./pcseg/data/dataset/nuscenes/nuscenes.yaml', 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        if self.train_val is False:
            with open(os.path.join(root_path, 'nuscenes_infos_%s_sweep.pkl'%self.split), 'rb') as f:
                data_sweep = pickle.load(f)
            self.nusc_infos_sweep = data_sweep['infos_sweep']
            self.global_indexes = data_sweep['global_indexes']
            self.local_indexes = data_sweep['local_indexes']
            self.scene_tokens = data_sweep['scene_tokens']
            self.token2samplelist = {}
            print(f'The total sweep is {len(self.nusc_infos_sweep)}')

        self._sample_idx = np.arange(len(self.nusc_infos))
        self.samples_per_epoch = self.data_cfgs.get('SAMPLES_PER_EPOCH', -1)
        if self.samples_per_epoch == -1 or not self.training:
            self.samples_per_epoch = len(self.nusc_infos)

        if self.training:
            self.resample()
        else:
            self.sample_idx = self._sample_idx

        self.multiscan      = self.data_cfgs.MULTISCAN
        self.step           = self.data_cfgs.STEP
        self.flexible_steps = self.data_cfgs.FLEXIBLE_STEPS
        self.pseudo_mask   = self.data_cfgs.PSEUDO_MASK

    def __len__(self):
        return len(self.sample_idx)

    def resample(self):
        self.sample_idx = np.random.choice(self._sample_idx, self.samples_per_epoch)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path'][16:]
        if self.data_path_ceph is not None:
            raw_data = np.copy(self.petrel_client.load_bin(os.path.join(self.data_path_ceph, lidar_path), dtype='float32').reshape([-1, 5]))
        else:
            raw_data = np.fromfile(os.path.join(self.root_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        raw_data[:, 4] = 0

        if self.split == 'test' and self.seq == -1:
            lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
            annotated_data = np.zeros((len(raw_data), 1))
            lidarseg_labels_filename = lidar_sd_token + '_lidarseg.bin'
        else:
            lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
            lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                    self.nusc.get('lidarseg', lidar_sd_token)['filename'])
            annotated_data = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        if self.nusc.get('sample', self.nusc_infos[index-1]['token'])['scene_token'] == \
                    self.nusc.get('sample', self.nusc_infos[index]['token'])['scene_token']:
            raw_data_ms, annotated_data_ms, annotated_data_ms_pseudo,  annotated_data_ms_mask \
                        = self.multiscan_fuse(index, lidar_sd_token, self.multiscan, self.step, self.flexible_steps)
            raw_data_ms = np.concatenate([raw_data, raw_data_ms[annotated_data_ms_mask]])
            annotated_data_ms = np.concatenate([annotated_data, annotated_data_ms[annotated_data_ms_mask]])
        else:
            raw_data_ms = raw_data
            annotated_data_ms = annotated_data

        prob = np.random.choice(2, 1)
        index_another = np.random.choice(len(self.nusc_infos))
        if self.augment == 'GlobalAugment_LP' or self.augment == 'GlobalAugment_L' or self.augment == 'GlobalAugment_P':
            if self.split == 'train' and (self.augment == 'GlobalAugment_LP' or self.augment == 'GlobalAugment_L') and prob == 1:
                info1 = self.nusc_infos[index_another]
                lidar_path1 = info1['lidar_path'][16:]
                if self.data_path_ceph is not None:
                    raw_data1 = np.copy(self.petrel_client.load_bin(os.path.join(self.data_path_ceph, lidar_path1), dtype='float32').reshape([-1, 5]))
                else:
                    raw_data1 = np.fromfile(os.path.join(self.root_path, lidar_path1), dtype=np.float32, count=-1).reshape([-1, 5])

                lidar_sd_token1 = self.nusc.get('sample', info1['token'])['data']['LIDAR_TOP']
                lidarseg_labels_filename1 = os.path.join(self.nusc.dataroot,
                                                        self.nusc.get('lidarseg', lidar_sd_token1)['filename'])
                annotated_data1 = np.fromfile(lidarseg_labels_filename1, dtype=np.uint8).reshape([-1, 1])
                annotated_data1 = np.vectorize(self.learning_map.__getitem__)(annotated_data1)

                assert len(annotated_data1) == len(raw_data1)

                if self.nusc.get('sample', self.nusc_infos[index_another-1]['token'])['scene_token'] == \
                            self.nusc.get('sample', self.nusc_infos[index_another]['token'])['scene_token']:
                    raw_data1_ms, annotated_data1_ms, annotated_data1_ms_pseudo,  annotated_data1_ms_mask \
                                = self.multiscan_fuse(index_another, lidar_sd_token1, self.multiscan, self.step, self.flexible_steps)
                    raw_data1_ms = np.concatenate([raw_data1, raw_data1_ms[annotated_data1_ms_mask]])
                    annotated_data1_ms = np.concatenate([annotated_data1, annotated_data1_ms[annotated_data1_ms_mask]])
                else:
                    raw_data1_ms = raw_data1
                    annotated_data1_ms = annotated_data1

                raw_data, annotated_data, strategy = lasermix_aug(raw_data, 
                        annotated_data, raw_data1, annotated_data1, return_strategy=True)
                raw_data_ms, annotated_data_ms, strategy_ms = lasermix_aug(raw_data_ms, 
                        annotated_data_ms, raw_data1_ms, annotated_data1_ms, strategy=strategy, return_strategy=True)

                assert strategy==strategy_ms
            
            elif self.split == 'train' and (self.augment == 'GlobalAugment_LP' or self.augment == 'GlobalAugment_P') and prob == 0:
                info1 = self.nusc_infos[index_another]
                lidar_path1 = info1['lidar_path'][16:]
                if self.data_path_ceph is not None:
                    raw_data1 = np.copy(self.petrel_client.load_bin(os.path.join(self.data_path_ceph, lidar_path1), dtype='float32').reshape([-1, 5]))
                else:
                    raw_data1 = np.fromfile(os.path.join(self.root_path, lidar_path1), dtype=np.float32, count=-1).reshape([-1, 5])

                lidar_sd_token1 = self.nusc.get('sample', info1['token'])['data']['LIDAR_TOP']
                lidarseg_labels_filename1 = os.path.join(self.nusc.dataroot,
                                                        self.nusc.get('lidarseg', lidar_sd_token1)['filename'])
                annotated_data1 = np.fromfile(lidarseg_labels_filename1, dtype=np.uint8).reshape([-1, 1])
                annotated_data1 = np.vectorize(self.learning_map.__getitem__)(annotated_data1)

                assert len(annotated_data1) == len(raw_data1)

                if self.nusc.get('sample', self.nusc_infos[index_another-1]['token'])['scene_token'] == \
                            self.nusc.get('sample', self.nusc_infos[index_another]['token'])['scene_token']:
                    raw_data1_ms, annotated_data1_ms, annotated_data1_ms_pseudo,  annotated_data1_ms_mask \
                                = self.multiscan_fuse(index_another, lidar_sd_token1, self.multiscan, self.step, self.flexible_steps)
                    raw_data1_ms = np.concatenate([raw_data1, raw_data1_ms[annotated_data1_ms_mask]])
                    annotated_data1_ms = np.concatenate([annotated_data1, annotated_data1_ms[annotated_data1_ms_mask]])
                else:
                    raw_data1_ms = raw_data1
                    annotated_data1_ms = annotated_data1

                alpha = (np.random.random() - 1) * np.pi
                beta = alpha + np.pi
                annotated_data1 = annotated_data1.reshape(-1)
                annotated_data = annotated_data.reshape(-1)
                raw_data, annotated_data, swap_flag, rotate_flag = polarmix(
                    raw_data, annotated_data, raw_data1, annotated_data1,
                    alpha=alpha, beta=beta, instance_classes=instance_classes, 
                    Omega=Omega, return_strategy=True
                )
                annotated_data = annotated_data.reshape(-1, 1)
        
                annotated_data1_ms = annotated_data1_ms.reshape(-1)
                annotated_data_ms = annotated_data_ms.reshape(-1)
                raw_data_ms, annotated_data_ms, swap_flag_ms, rotate_flag_ms = polarmix(
                    raw_data_ms, annotated_data_ms, raw_data1_ms, annotated_data1_ms,
                    alpha=alpha, beta=beta, instance_classes=instance_classes, 
                    Omega=Omega, swap_flag=swap_flag, rotate_flag=rotate_flag, return_strategy=True
                )
                annotated_data_ms = annotated_data_ms.reshape(-1, 1)
                assert swap_flag==swap_flag_ms
                assert rotate_flag==rotate_flag_ms
                
        pc_data = {
            'xyzret': raw_data,
            'xyzret_ms': raw_data_ms,
            'labels': annotated_data.astype(np.uint8),
            'labels_ms': annotated_data_ms.astype(np.uint8),
            'path': lidarseg_labels_filename,
        }

        return pc_data

    def multiscan_fuse(self, index, lidar_sd_token, multiscan, step, flexible_steps):
        global_index = self.global_indexes[index]
        info0 = self.nusc_infos_sweep[global_index]
        scene_token0 = self.scene_tokens[global_index]
        raw_data_ms = []
        annotated_data_ms = []
        annotated_data_ms_pseudo = []
        annotated_data_ms_mask = []

        if lidar_sd_token in self.token2samplelist.keys():
            sample_list = self.token2samplelist[lidar_sd_token]
        else:
            delta_idx = 0
            total_list = []
            dist_list = []
            while(len(dist_list) == 0 or dist_list[-1] <= multiscan * step):
                delta_idx -= 1

                info = self.nusc_infos_sweep[global_index+delta_idx]
                scene_token = self.scene_tokens[global_index+delta_idx]
                if scene_token != scene_token0:
                    dist_list.append(1000)
                    break

                origin = np.zeros((1,5), dtype=np.float)
                if 'data_path' in info.keys():
                    origin[:, :3] = origin[:, :3] @ info[
                        'sensor2lidar_rotation'].T
                    origin[:, :3] += info['sensor2lidar_translation']
                if self.local_indexes[global_index+delta_idx] != index:
                    info_father = self.nusc_infos[self.local_indexes[global_index+delta_idx]]
                    origin = self.transform_point(origin, info0, info_father)

                total_list.append(delta_idx)
                dist_list.append(np.linalg.norm(origin.reshape(-1)[:2], ord = 2))

            cur_scan = 1
            sample_list = []
            # assert len(total_list) != 0
            for idx in range(len(total_list)):
                if dist_list[idx] - cur_scan * step > 0 or ((dist_list[idx] < dist_list[idx+1]) and (np.abs(dist_list[idx] - cur_scan * step) < np.abs(dist_list[idx+1] - cur_scan * step))):
                    sample_list.append(total_list[idx])
                    cur_scan += 1
                if cur_scan > multiscan:
                    break
            for delta_idx in total_list:
                if 'lidar_path' in self.nusc_infos_sweep[global_index + delta_idx].keys():
                    sample_list.append(delta_idx)
            sample_list = list(set(sample_list))
            sample_list.sort()
            self.token2samplelist[lidar_sd_token] = sample_list

        # assert len(sample_list) != 0
        for idx, delta_idx in enumerate(sample_list):
            info = self.nusc_infos_sweep[global_index + delta_idx]

            if 'lidar_path' in info.keys():
                lidar_path = info['lidar_path'][16:]
                if self.data_path_ceph is not None:
                    raw_data = np.copy(self.petrel_client.load_bin(os.path.join(self.data_path_ceph, lidar_path), dtype='float32').reshape([-1, 5]))
                else:
                    raw_data = np.fromfile(os.path.join(self.root_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
                no_ego = ~((np.abs(raw_data[:, 0]) < 1.0) & (np.abs(raw_data[:, 1]) < 1.5))
                raw_data[:, 4] = info0['timestamp'] / 1e6 - info['timestamp'] / 1e6
                raw_data = self.transform_point(raw_data, info0, info)

                try:
                    lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
                    lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                            self.nusc.get('lidarseg', lidar_sd_token)['filename'])
                    annotated_data = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
                    annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
                except:
                    lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
                    annotated_data = np.zeros((raw_data.shape[0], 1), dtype=np.uint8)
            else:
                if self.data_path_ceph is not None:
                    points_sweep = np.copy(self.petrel_client.load_bin(os.path.join(self.data_path_ceph, info['data_path'][16:]), dtype='float32').reshape([-1, 5]))
                else:
                    points_sweep = np.fromfile(os.path.join(self.root_path, info['data_path'][16:]), dtype=np.float32, count=-1).reshape([-1, 5])
                no_ego = ~((np.abs(points_sweep[:, 0]) < 1.0) & (np.abs(points_sweep[:, 1]) < 1.5))
                points_sweep[:, :3] = points_sweep[:, :3] @ info[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += info['sensor2lidar_translation']
                points_sweep[:, 4] = info0['timestamp'] / 1e6 - info['timestamp'] / 1e6
                raw_data = points_sweep

                if self.local_indexes[global_index+delta_idx] != index:
                    info_father = self.nusc_infos[self.local_indexes[global_index+delta_idx]]
                    raw_data = self.transform_point(raw_data, info0, info_father)

                lidar_sd_token = info['sample_data_token']
                annotated_data = np.zeros((raw_data.shape[0], 1), dtype=np.uint8)

            if self.pseudo_mask == 'mink_sweep_notta':
                pseudo_mask_path = '/YourHome/PCSeg/logs/voxel/nuscenes/minkunet_mk34_cr10/default/results/lidarseg/trainval_sweep_notta'

            lidarseg_labels_filename_pseudo = os.path.join(pseudo_mask_path, lidar_sd_token + '_lidarseg.bin')
            annotated_data_pseudo = np.fromfile(lidarseg_labels_filename_pseudo, dtype=np.uint8).reshape([-1, 1])

            raw_data = raw_data[no_ego]
            annotated_data = annotated_data[no_ego]
            annotated_data_pseudo = annotated_data_pseudo[no_ego]
            annotated_data_mask = np.zeros(len(annotated_data_pseudo), dtype=np.bool)
            for class_idx, class_step in enumerate(self.flexible_steps):
                if class_step == 0:
                    continue
                if (idx + 1) % class_step == 0:
                    annotated_data_mask = annotated_data_mask | (annotated_data_pseudo[:,0] == class_idx)

            raw_data_ms.append(raw_data)
            annotated_data_ms.append(annotated_data)
            annotated_data_ms_pseudo.append(annotated_data_pseudo)
            annotated_data_ms_mask.append(annotated_data_mask)

        raw_data_ms = np.concatenate(raw_data_ms, 0)
        annotated_data_ms = np.concatenate(annotated_data_ms, 0)
        annotated_data_ms_pseudo = np.concatenate(annotated_data_ms_pseudo, 0)
        annotated_data_ms_mask = np.concatenate(annotated_data_ms_mask, 0)

        return raw_data_ms, annotated_data_ms, annotated_data_ms_pseudo,  annotated_data_ms_mask

    def transform_point(self, raw_data, info0, info):

        l2e_r = info0['lidar2ego_rotation']
        l2e_t = info0['lidar2ego_translation']
        e2g_r = info0['ego2global_rotation']
        e2g_t = info0['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        l2e_r_s = info['lidar2ego_rotation']
        l2e_t_s = info['lidar2ego_translation']
        e2g_r_s = info['ego2global_rotation']
        e2g_t_s = info['ego2global_translation']
        l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
        e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

        R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                    ) + l2e_t @ np.linalg.inv(l2e_r_mat).T

        raw_data[:, :3] = raw_data[:, :3] @ R + T

        return raw_data

    @staticmethod
    def collate_batch(batch_list):
        raise NotImplementedError