import os
import numpy as np
from torch.utils import data
from .LaserMix_nuscenes import lasermix_aug, lasermix_aug_
from .PolarMix_nuscenes import polarmix
from pyquaternion import Quaternion
from typing import List, Tuple, Union
from itertools import repeat
from PIL import Image
import random
import pickle
import yaml
import pdb
import copy
from nuscenes.utils.geometry_utils import view_points
from random import sample

# used for polarmix
instance_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]

class NuscenesMsMmDataset(data.Dataset):
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
        self.debug = data_cfgs.get('DEBUG', False)

        from pcseg.data.dataset.ceph import PetrelBackend
        self.petrel_client = PetrelBackend()
        self.data_path_ceph = data_cfgs.get('DATA_PATH_CEPH', None) # "cluster5:s3://wxp-DataSets/nuscenes/"

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

        self.img_view = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
                         'CAM_FRONT_LEFT']
        self.used_view = data_cfgs.get('USED_VIEW')
        self.get_path_infos_cam_lidar()
        self.resize = 0.5

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
            self.token2samplelist_fov = {}
            print(f'The total sweep is {len(self.nusc_infos_sweep)}')

        if self.debug:
            self.nusc_infos = self.nusc_infos[:100]

        self._sample_idx = np.arange(len(self.nusc_infos))
        # self._sample_idx = self._sample_idx[600*8:]
        self.samples_per_epoch = self.data_cfgs.get('SAMPLES_PER_EPOCH', -1)
        if self.samples_per_epoch == -1 or not self.training:
            self.samples_per_epoch = len(self.nusc_infos)

        if self.training:
            self.resample()
        else:
            self.sample_idx = self._sample_idx

        self.multiscan      = self.data_cfgs.MULTISCAN
        self.multiscan_image = self.data_cfgs.MULTISCAN_IMAGE
        self.step_image = self.data_cfgs.STEP_IMAGE
        self.multiscan_interval = self.data_cfgs.MULTISCAN_INTERVAL
        self.height = self.data_cfgs.HEIGHT
        self.width = self.data_cfgs.WIDTH
        self.image_jitter = self.data_cfgs.IMAGE_JITTER if self.training else False
        self.image_flip = self.data_cfgs.IMAGE_FLIP if self.training else False
        self.step           = self.data_cfgs.STEP
        self.flexible_steps = self.data_cfgs.FLEXIBLE_STEPS
        self.pseudo_mask   = self.data_cfgs.PSEUDO_MASK
        self.paint_dist = self.data_cfgs.PAINT_DIST

    def get_path_infos_cam_lidar(self):
        self.token_list = []

        for info in  self.nusc_infos:
            sample = self.nusc.get('sample', info['token'])
            scene_token = sample['scene_token']
            lidar_token = sample['data']['LIDAR_TOP']  # 360 lidar

            cam_token = []
            for i in self.img_view:
                cam_token.append(sample['data'][i])
            self.token_list.append(
                {'lidar_token': lidar_token,
                    'cam_token': cam_token}
            )

    def __len__(self):
        return len(self.sample_idx)

    def resample(self):
        self.sample_idx = np.random.choice(self._sample_idx, self.samples_per_epoch)

    def __getitem__(self, index):
        # index = self.sample_idx[index]
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

        raw_data_fov_ms, annotated_data_fov_ms, image_ms, depth_map_ms, lidar_map_ms, semantic_map_ms \
            = self.multiscan_fuse_fov(index)

        pc_data = {
            'xyzret': raw_data,
            'xyzret_ms': raw_data_ms,

            'xyzret_fov_ms': raw_data_fov_ms.astype(np.float32),
            'labels_fov_ms': annotated_data_fov_ms.astype(np.uint8),
            'image_ms': image_ms,
            'depth_map_ms': depth_map_ms,
            'lidar_map_ms': lidar_map_ms,
            'semantic_map_ms': semantic_map_ms,

            'labels': annotated_data.astype(np.uint8),
            'labels_ms': annotated_data_ms.astype(np.uint8),
            'path': lidarseg_labels_filename,
        }

        return pc_data

    def multiscan_fuse_fov(self, index):
        info0 = self.nusc_infos[index]
        lidar_sd_token0 = self.nusc.get('sample', info0['token'])['data']['LIDAR_TOP']
        raw_data_fov_ms = []
        annotated_data_fov_ms = []
        cropped_image_ms = []
        cropped_depth_map_ms = []
        cropped_lidar_map_ms = []
        cropped_semantic_map_ms = []

        if self.multiscan_image == 0:
            sample_list = [0]
        elif lidar_sd_token0 in self.token2samplelist_fov.keys():
            sample_list = self.token2samplelist_fov[lidar_sd_token0]
        else:
            delta_idx = 0
            total_list = []
            dist_list = []
            while(len(dist_list) == 0 or dist_list[-1] <= self.multiscan_image * self.step_image):
                delta_idx -= 1
                info = self.nusc_infos[index + delta_idx]
                if self.nusc.get('sample', info['token'])['scene_token'] != \
                    self.nusc.get('sample', info0['token'])['scene_token']:
                    dist_list.append(1000)
                    break
                origin = np.zeros((1,5), dtype=np.float)
                origin = self.transform_point(origin, info0, info)
                total_list.append(delta_idx)
                dist_list.append(np.linalg.norm(origin.reshape(-1)[:2], ord = 2))
            cur_scan = 1
            sample_list = []
            abandom_list = []
            for idx in range(len(total_list)):
                if dist_list[idx] - cur_scan * self.step_image > 0 or ((dist_list[idx] < dist_list[idx+1]) and \
                    (np.abs(dist_list[idx] - cur_scan * self.step_image) < np.abs(dist_list[idx+1] - cur_scan * self.step_image))):
                    sample_list.append(total_list[idx])
                    cur_scan += 1
                else:
                    abandom_list.append(total_list[idx])
                if cur_scan > self.multiscan_image:
                    break
            if (len(sample_list) < self.multiscan_image) and (len(abandom_list) > 0):
                sample_list += sample(abandom_list, min(self.multiscan_image-len(sample_list), len(abandom_list)))

            sample_list = list(set(sample_list))
            sample_list.append(0)
            sample_list.sort()
            self.token2samplelist_fov[lidar_sd_token0] = sample_list

        # print('len(sample_list): ', len(sample_list))
        for batch_idx, delta_idx in enumerate(sample_list):
            if (index + delta_idx <0) or (index + delta_idx >= len(self.nusc_infos)) or \
                    (self.nusc.get('sample', self.nusc_infos[index+delta_idx]['token'])['scene_token'] != \
                    self.nusc.get('sample', self.nusc_infos[index]['token'])['scene_token']):
                continue

            info = self.nusc_infos[index + delta_idx]
            lidar_path = info['lidar_path'][16:]
            if self.data_path_ceph is not None:
                raw_data = np.copy(self.petrel_client.load_bin(os.path.join(self.data_path_ceph, lidar_path), dtype='float32').reshape([-1, 5]))
            else:
                raw_data = np.fromfile(os.path.join(self.root_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
            no_ego = ~((np.abs(raw_data[:, 0]) < 1.0) & (np.abs(raw_data[:, 1]) < 1.5))
            raw_data[:, 4] = info0['timestamp'] / 1e6 - info['timestamp'] / 1e6
            lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
            try:
                lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                        self.nusc.get('lidarseg', lidar_sd_token)['filename'])
                annotated_data = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
                annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
            except:
                # print('use empty annotated_data')
                annotated_data = np.zeros((raw_data.shape[0], 1), dtype=np.uint8)
            raw_data = raw_data[no_ego]
            annotated_data = annotated_data[no_ego]

            for meta_delta_idx in range(-self.multiscan_interval, 0):
                if (index + delta_idx + meta_delta_idx <0) or \
                    (index + delta_idx + meta_delta_idx >= len(self.nusc_infos)) or \
                    (self.nusc.get('sample', self.nusc_infos[index+delta_idx+meta_delta_idx]['token'])['scene_token'] != \
                    self.nusc.get('sample', self.nusc_infos[index]['token'])['scene_token']):
                    continue
                if (batch_idx>0) and (delta_idx+meta_delta_idx <= sample_list[batch_idx-1]):
                    continue
                info1 = self.nusc_infos[index + delta_idx + meta_delta_idx]
                lidar_path1 = info1['lidar_path'][16:]
                if self.data_path_ceph is not None:
                    raw_data1 = np.copy(self.petrel_client.load_bin(os.path.join(self.data_path_ceph, lidar_path1), dtype='float32').reshape([-1, 5]))
                else:
                    raw_data1 = np.fromfile(os.path.join(self.root_path, lidar_path1), dtype=np.float32, count=-1).reshape([-1, 5])
                no_ego1 = ~((np.abs(raw_data1[:, 0]) < 1.0) & (np.abs(raw_data1[:, 1]) < 1.5))
                raw_data1[:, 4] = info0['timestamp'] / 1e6 - info1['timestamp'] / 1e6
                lidar_sd_token1 = self.nusc.get('sample', info1['token'])['data']['LIDAR_TOP']
                try:
                    lidarseg_labels_filename1 = os.path.join(self.nusc.dataroot,
                                                            self.nusc.get('lidarseg', lidar_sd_token1)['filename'])
                    annotated_data1 = np.fromfile(lidarseg_labels_filename1, dtype=np.uint8).reshape([-1, 1])
                    annotated_data1 = np.vectorize(self.learning_map.__getitem__)(annotated_data1)
                except:
                    # print('use empty annotated_data')
                    annotated_data1 = np.zeros((raw_data1.shape[0], 1), dtype=np.uint8)
                raw_data1 = raw_data1[no_ego1]
                annotated_data1 = annotated_data1[no_ego1]
                raw_data1 = self.transform_point(raw_data1, info, info1)
                # print('batch %d increase %d frame, %d points' % (batch_idx, meta_delta_idx, len(raw_data1)))
                raw_data = np.concatenate([raw_data, raw_data1], axis=0)
                annotated_data = np.concatenate([annotated_data, annotated_data1], axis=0)

            if self.paint_dist > 0:
                radius = np.linalg.norm(raw_data[:, :2], ord=2, axis=1, keepdims=False)
                raw_data = raw_data[radius <= self.paint_dist]
                annotated_data = annotated_data[radius <= self.paint_dist]
            for view_idx, image_id in enumerate(self.used_view):
                raw_data_fov, annotated_data_fov, cropped_image, cropped_depth_map, cropped_lidar_map, cropped_semantic_map \
                    = self.get_fov_points(index + delta_idx, image_id, batch_idx * len(self.used_view) + view_idx, \
                        copy.deepcopy(raw_data[:, :4]), copy.deepcopy(annotated_data), lidar_sd_token)
                raw_data_fov = self.transform_point(raw_data_fov, info0, info)

                raw_data_fov_ms.append(raw_data_fov)
                annotated_data_fov_ms.append(annotated_data_fov)
                cropped_image_ms.append(cropped_image)
                cropped_depth_map_ms.append(cropped_depth_map)
                cropped_lidar_map_ms.append(cropped_lidar_map)
                cropped_semantic_map_ms.append(cropped_semantic_map)

        raw_data_fov_ms = np.concatenate(raw_data_fov_ms, 0)
        annotated_data_fov_ms = np.concatenate(annotated_data_fov_ms, 0)
        cropped_image_ms = np.stack(cropped_image_ms, axis=0)
        cropped_depth_map_ms = np.stack(cropped_depth_map_ms, axis=0)
        cropped_lidar_map_ms = np.stack(cropped_lidar_map_ms, axis=0)
        cropped_semantic_map_ms = np.stack(cropped_semantic_map_ms, axis=0)

        return raw_data_fov_ms, annotated_data_fov_ms, cropped_image_ms, \
            cropped_depth_map_ms, cropped_lidar_map_ms, cropped_semantic_map_ms

    def get_fov_points(self, index, image_id, img_batch, raw_data, annotated_data, lidar_sd_token):
        cam_view = self.img_view[image_id]
        cam_sample_token = self.token_list[index]['cam_token'][image_id]
        image_file = os.path.join(self.nusc.dataroot, self.nusc.get('sample_data', cam_sample_token)['filename'])
        image = Image.open(image_file)
        im_shape = (image.size[1], image.size[0])

        view = '' if cam_view == 'CAM_FRONT' else cam_view.replace('CAM', '')
        if self.data_path_ceph is not None:
            depth_map = np.zeros((image.size[1], image.size[0], 1), dtype=np.float32)
            lidar_map = np.zeros((image.size[1], image.size[0], 4), dtype=np.float32)
            # depth_map = self.petrel_client.load_np(image_file.replace(cam_view, 'DEPTH_MAP%s'%view).replace('.jpg', '.npy').replace(self.root_path, self.data_path_ceph))
            # lidar_map = self.petrel_client.load_np(image_file.replace(cam_view, 'LIDAR_MAP%s'%view).replace('.jpg', '.npy').replace(self.root_path, self.data_path_ceph))
            semantic_map = self.petrel_client.load_np(image_file.replace(cam_view, 'SEMANTIC_MAP_SAM%s'%view).replace('.jpg', '.npy').replace(self.root_path, self.data_path_ceph))
        else:
            depth_map = np.zeros((image.size[1], image.size[0], 1), dtype=np.float32)
            lidar_map = np.zeros((image.size[1], image.size[0], 4), dtype=np.float32)
            # depth_map = np.load(image_file.replace(cam_view, 'DEPTH_MAP%s'%view).replace('.jpg', '.npy'))
            # lidar_map = np.load(image_file.replace(cam_view, 'LIDAR_MAP%s'%view).replace('.jpg', '.npy'))
            semantic_map = np.load(image_file.replace(cam_view, 'SEMANTIC_MAP_SAM%s'%view).replace('.jpg', '.npy'))

        pc = raw_data[:, :3].copy().T
        pointsensor = self.nusc.get('sample_data', lidar_sd_token)
        cs_record_lidar = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pose_record_lidar = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        
        cam_path, boxes_front_cam, cam_intrinsic = self.nusc.get_sample_data(cam_sample_token)
        cam = self.nusc.get('sample_data', cam_sample_token)
        cs_record_cam = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pose_record_cam = self.nusc.get('ego_pose', cam['ego_pose_token'])

        pc = Quaternion(cs_record_lidar['rotation']).rotation_matrix @ pc
        pc = pc + np.array(cs_record_lidar['translation'])[:, np.newaxis]
        pc = Quaternion(pose_record_lidar['rotation']).rotation_matrix @ pc
        pc = pc + np.array(pose_record_lidar['translation'])[:, np.newaxis]
        pc = pc - np.array(pose_record_cam['translation'])[:, np.newaxis]
        pc = Quaternion(pose_record_cam['rotation']).rotation_matrix.T @ pc
        pc = pc - np.array(cs_record_cam['translation'])[:, np.newaxis]
        pc = Quaternion(cs_record_cam['rotation']).rotation_matrix.T @ pc
        depths = pc[2, :]
        points = view_points(pc, np.array(cam_intrinsic), normalize=True)
        points = points.astype(np.float32)
        keep_mask = np.ones(depths.shape[0], dtype=bool)
        keep_mask = np.logical_and(keep_mask, depths > 0)
        keep_mask = np.logical_and(keep_mask, points[0, :] > 0)
        keep_mask = np.logical_and(keep_mask, points[0, :] < im_shape[1])
        keep_mask = np.logical_and(keep_mask, points[1, :] > 0)
        keep_mask = np.logical_and(keep_mask, points[1, :] < im_shape[0])

        raw_data_frustum_uv = points.T[:, :2][keep_mask].astype(np.int)
        raw_data_frustum_uv = np.ascontiguousarray(np.fliplr(raw_data_frustum_uv))

        H, W = im_shape[0], im_shape[1]
        raw_data_frustum_uv[:, 0] = np.floor(self.resize * raw_data_frustum_uv[:, 0])
        raw_data_frustum_uv[:, 1] = np.floor(self.resize * raw_data_frustum_uv[:, 1])
        cropped_image = image.resize((int(W*self.resize), int(H*self.resize)), Image.BILINEAR)
        cropped_image = np.array(cropped_image, dtype=np.float32, copy=False) / 255.
        cropped_image = cropped_image[...,[2,1,0]]
        cropped_image = cropped_image[2:, ...] # 900 * 0.5 = 450 --> 448
        cropped_depth_map = depth_map[2:, ...]
        cropped_lidar_map = lidar_map[2:, ...]
        cropped_semantic_map = semantic_map[2:, ...]

        crop_mask = raw_data_frustum_uv[:, 0] >= 2
        keep_mask[keep_mask] = crop_mask
        raw_data_frustum_uv = raw_data_frustum_uv[crop_mask]
        raw_data_frustum_uv[:, 0] -= 2

        raw_data_frustum_uv[:, 0] += (self.height * img_batch)
        raw_data = np.concatenate([raw_data[keep_mask], raw_data_frustum_uv], axis=-1)
        annotated_data = annotated_data[keep_mask]

        return raw_data, annotated_data, cropped_image, cropped_depth_map, cropped_lidar_map, cropped_semantic_map

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