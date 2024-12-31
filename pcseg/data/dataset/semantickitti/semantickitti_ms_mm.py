import os
import numpy as np
from torch.utils import data
from .semantickitti_utils import LEARNING_MAP, LEARNING_MAP_INV
from .LaserMix_semantickitti import lasermix_aug
from .PolarMix_semantickitti import polarmix
import random
from typing import List, Tuple, Union
from itertools import repeat
from pcseg.data.dataset.ceph import PetrelBackend
import copy
from PIL import Image
import mmcv

# used for polarmix
instance_classes = [1, 2, 3, 4, 5, 6, 7, 8]
Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]

def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


class SemantickittiMsMmDataset(data.Dataset):
    def __init__(
        self,
        data_cfgs=None,
        training: bool = True,
        class_names: list = None,
        root_path: str = None,
        logger = None,
        if_scribble: bool = False,
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
        self.dynamic_step = data_cfgs.get('DYNAMIC_STEP', False)
        self.augment = data_cfgs.AUGMENT
        self.if_scribble = if_scribble

        if self.training:
            self.split = 'train'
        else:
            self.split = 'val'
        if self.tta:
            self.split = 'test'

        self.trainval_seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        self.test_seqs = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

        if self.split == 'train':
            if self.train_val:
                self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
            else:
                self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'test':
            if self.seq == -1:
                self.seqs = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
            elif self.seq == -2:
                self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
            elif self.seq == -3:
                self.seqs = ['08']
        else:
            raise Exception('split must be train/val/train_val/test.')
        
        self.annos = []
        for seq in self.seqs:
            self.annos += absoluteFilePaths('/'.join([self.root_path, str(seq).zfill(2), 'velodyne']))
        self.annos.sort()
        self.annos_another = self.annos.copy()
        random.shuffle(self.annos_another)
        print(f'The total sample is {len(self.annos)}')

        self._sample_idx = np.arange(len(self.annos))

        self.samples_per_epoch = self.data_cfgs.get('SAMPLES_PER_EPOCH', -1)
        if self.samples_per_epoch == -1 or not self.training:
            self.samples_per_epoch = len(self.annos)

        if self.training:
            self.resample()
        else:
            self.sample_idx = self._sample_idx

        self.multiscan_scale = data_cfgs.get('MULTISCAN_SCALE', 1.0)
        self.multiscan      = int(self.data_cfgs.MULTISCAN / self.multiscan_scale)
        self.multiscan_image = int(self.data_cfgs.MULTISCAN_IMAGE / self.multiscan_scale)
        self.step_image = int(self.data_cfgs.STEP_IMAGE / self.multiscan_scale)
        self.height = self.data_cfgs.HEIGHT
        self.width = self.data_cfgs.WIDTH
        self.image_jitter = self.data_cfgs.IMAGE_JITTER if self.training else False
        self.image_flip = self.data_cfgs.IMAGE_FLIP if self.training else False
        self.flip_ratio = 0.5
        self.brightness_delta = 32
        self.contrast_lower, self.contrast_upper = 0.5, 1.5
        self.saturation_lower, self.saturation_upper = 0.5, 1.5
        self.hue_delta = 18
        self.only_history    = self.data_cfgs.ONLY_HISTORY
        self.pseudo_mask   = self.data_cfgs.PSEUDO_MASK
        self.flexible_steps = [int(class_step / self.multiscan_scale)  for class_step in self.data_cfgs.FLEXIBLE_STEPS]
        self.fov_dist = self.data_cfgs.FOV_DIST
        print('====================================')
        print('multiscan: %d' % self.multiscan)
        print('multiscan_image: %d' % self.multiscan_image)
        print('step_image: %d' % self.step_image)
        print('flexible_steps: ',self.flexible_steps)
        print('====================================')

        self.petrel_client = PetrelBackend()

        self.calibrations = []
        self.times = []
        self.poses = []
        self.load_calib_poses()

    def __len__(self):
        return len(self.sample_idx)

    def resample(self):
        self.sample_idx = np.random.choice(self._sample_idx, self.samples_per_epoch)
    
    def get_kitti_points_ringID(self, points):
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        yaw = -np.arctan2(scan_y, -scan_x)
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
        proj_y = np.zeros_like(proj_x)
        proj_y[new_raw] = 1
        ringID = np.cumsum(proj_y)
        ringID = np.clip(ringID, 0, 63)
        return ringID

    def __getitem__(self, index):
        raw_data = np.fromfile(self.annos[index], dtype=np.float32).reshape((-1, 4))

        if self.split == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            if self.if_scribble:  # ScribbleKITTI (weak label)
                annos = self.annos[index].replace('SemanticKITTI', 'ScribbleKITTI')
                annotated_data = np.fromfile(
                    annos.replace('velodyne', 'scribbles')[:-3] + 'label', dtype=np.uint32
                ).reshape((-1, 1))
            else:  # SemanticKITTI (full label)
                if self.annos[index].split('/sequences/')[-1][:2] in self.trainval_seqs:
                    annotated_data = np.fromfile(
                        self.annos[index].replace('velodyne', 'labels')[:-3] + 'label', dtype=np.uint32
                    ).reshape((-1, 1))
            
            annotated_data = annotated_data & 0xFFFF
            annotated_data = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data)

        if int(self.annos[index][-10:-4]) - 1 >=0:
            raw_data_ms, annotated_data_ms, annotated_data_ms_mask, \
                raw_data_fov_ms, image_ms, depth_map_ms, lidar_map_ms, semantic_map_ms \
                = self.multiscan_fuse(self.annos, index, self.multiscan, self.flexible_steps, self.multiscan_image, self.step_image)
            raw_data_ms = np.concatenate([raw_data, raw_data_ms[annotated_data_ms_mask]])
            raw_data_ms = self.append_time_flag(raw_data, raw_data_ms)
            annotated_data_ms = np.concatenate([annotated_data, annotated_data_ms[annotated_data_ms_mask]])
        else:
            _, _, _, \
                raw_data_fov_ms, image_ms, depth_map_ms, lidar_map_ms, semantic_map_ms \
                = self.multiscan_fuse(self.annos, index, self.multiscan, self.flexible_steps, self.multiscan_image, self.step_image)
            raw_data_ms = raw_data
            raw_data_ms = self.append_time_flag(raw_data, raw_data_ms)
            annotated_data_ms = annotated_data

        prob = np.random.choice(2, 1)
        if self.augment == 'GlobalAugment_LP':
            if self.split == 'train' and prob == 1:
                raw_data1 = np.fromfile(self.annos_another[index], dtype=np.float32).reshape((-1, 4))

                if self.if_scribble:  # ScribbleKITTI (weak label)
                    annos1 = self.annos_another[index].replace('SemanticKITTI', 'ScribbleKITTI')
                    annotated_data1 = np.fromfile(
                        annos1.replace('velodyne', 'scribbles')[:-3] + 'label', dtype=np.uint32
                    ).reshape((-1, 1))
                else: # SemanticKITTI (full label)
                    if self.annos_another[index].split('/sequences/')[-1][:2] in self.trainval_seqs:
                        annotated_data1 = np.fromfile(
                            self.annos_another[index].replace('velodyne', 'labels')[:-3] + 'label', dtype=np.uint32
                        ).reshape((-1, 1))
                
                annotated_data1 = annotated_data1 & 0xFFFF
                annotated_data1 = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data1)
                assert len(annotated_data1) == len(raw_data1)

                if int(self.annos_another[index][-10:-4]) - 1 >=0:
                    raw_data1_ms, annotated_data1_ms,  annotated_data1_ms_mask \
                                = self.multiscan_fuse(self.annos_another, index, self.multiscan, self.flexible_steps)
                    raw_data1_ms = np.concatenate([raw_data1, raw_data1_ms[annotated_data1_ms_mask]])
                    raw_data1_ms = self.append_time_flag(raw_data1, raw_data1_ms)
                    annotated_data1_ms = np.concatenate([annotated_data1, annotated_data1_ms[annotated_data1_ms_mask]])
                else:
                    raw_data1_ms = raw_data1
                    raw_data1_ms = self.append_time_flag(raw_data1, raw_data1_ms)
                    annotated_data1_ms = annotated_data1

                raw_data, annotated_data, strategy = lasermix_aug(raw_data, 
                        annotated_data, raw_data1, annotated_data1, return_strategy=True)
                raw_data_ms, annotated_data_ms, strategy_ms = lasermix_aug(raw_data_ms, 
                        annotated_data_ms, raw_data1_ms, annotated_data1_ms, strategy=strategy, return_strategy=True)
                assert strategy==strategy_ms
            
            elif self.split == 'train' and prob == 0:
                raw_data1 = np.fromfile(self.annos_another[index], dtype=np.float32).reshape((-1, 4))

                if self.if_scribble:  # ScribbleKITTI (weak label)
                    annos1 = self.annos_another[index].replace('SemanticKITTI', 'ScribbleKITTI')
                    annotated_data1 = np.fromfile(
                        annos1.replace('velodyne', 'scribbles')[:-3] + 'label', dtype=np.uint32
                    ).reshape((-1, 1))
                else: # SemanticKITTI (full label)
                    if self.annos_another[index].split('/sequences/')[-1][:2] in self.trainval_seqs:
                        annotated_data1 = np.fromfile(
                            self.annos_another[index].replace('velodyne', 'labels')[:-3] + 'label', dtype=np.uint32
                        ).reshape((-1, 1))
                
                annotated_data1 = annotated_data1 & 0xFFFF
                annotated_data1 = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data1)
                assert len(annotated_data1) == len(raw_data1)

                if int(self.annos_another[index][-10:-4]) - 1 >=0:
                    raw_data1_ms, annotated_data1_ms,  annotated_data1_ms_mask \
                                = self.multiscan_fuse(self.annos_another, index, self.multiscan, self.flexible_steps)
                    raw_data1_ms = np.concatenate([raw_data1, raw_data1_ms[annotated_data1_ms_mask]])
                    raw_data1_ms = self.append_time_flag(raw_data1, raw_data1_ms)
                    annotated_data1_ms = np.concatenate([annotated_data1, annotated_data1_ms[annotated_data1_ms_mask]])
                else:
                    raw_data1_ms = raw_data1
                    raw_data1_ms = self.append_time_flag(raw_data1, raw_data1_ms)
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
        
        ringID = self.get_kitti_points_ringID(raw_data).reshape((-1, 1))
        raw_data= np.concatenate([raw_data, ringID.reshape(-1, 1)], axis=1).astype(np.float32)
        ringID_ms = self.get_kitti_points_ringID(raw_data_ms).reshape((-1,1))
        raw_data_ms= np.concatenate([raw_data_ms, ringID_ms.reshape(-1, 1)], axis=1).astype(np.float32)
        raw_data_fov_ms = raw_data_fov_ms.astype(np.float32)
        pc_data = {
            'xyzret': raw_data,
            'xyzret_ms': raw_data_ms,

            'xyzret_fov_ms': raw_data_fov_ms,
            'image_ms': image_ms,
            'depth_map_ms': depth_map_ms,
            'lidar_map_ms': lidar_map_ms,
            'semantic_map_ms': semantic_map_ms,

            'labels': annotated_data.astype(np.uint8),
            'labels_ms': annotated_data_ms.astype(np.uint8),
            'path': self.annos[index],
        }

        return pc_data

    def append_time_flag(self, raw_data, raw_data_ms):
        time_flag = np.zeros((len(raw_data_ms), 1), dtype=raw_data_ms.dtype)
        time_flag[:len(raw_data), 0] = 1
        raw_data_ms = np.concatenate([raw_data_ms[:, :4], time_flag, raw_data_ms[:, 4:]], axis=1)
        return raw_data_ms

    @staticmethod
    def collate_batch(batch_list):
        raise NotImplementedError

    def get_driving_dist(self, pose0, pose):
        origin = np.zeros((1,5), dtype=np.float64)
        origin_ = self.fuse_multi_scan(origin.copy(), pose0, pose)
        driving_dist = np.linalg.norm((origin - origin_).reshape(-1)[:2], ord = 2)
        return driving_dist

    def multiscan_fuse(self, annos, index, multiscan, flexible_steps, multiscan_image, step_image):
        raw_data_ms = []
        annotated_data_ms = []
        annotated_data_ms_mask = []

        raw_data_fov_ms = []
        cropped_image_ms = []
        cropped_depth_map_ms = []
        cropped_lidar_map_ms = []
        cropped_semantic_map_ms = []

        number_idx = int(annos[index][-10:-4])
        dir_idx = int(annos[index][-22:-20])
        pose0 = self.poses[dir_idx][number_idx]
        pose = self.poses[dir_idx][number_idx-1]
        speed = int(self.get_driving_dist(pose0, pose)  * 10)
        if self.dynamic_step and (speed > 10):
            print('speed %d' % speed)
            multiscan = multiscan//2
            flexible_steps = [class_step//2  for class_step in flexible_steps]
            multiscan_image = multiscan_image//2
            step_image = step_image//2

        assert multiscan_image > multiscan
        for delta_idx in range(-multiscan_image, multiscan):
            if self.only_history and delta_idx > 0:
                continue
            if (delta_idx < -multiscan) and (np.abs(delta_idx) % step_image !=0):
                continue
            try:
                pose = self.poses[dir_idx][number_idx + delta_idx]
                newpath = annos[index][:-10] + str(number_idx + delta_idx).zfill(6) + annos[index][-4:]
                raw_data = np.fromfile(newpath, dtype=np.float32).reshape((-1, 4))
            except:
                continue

            if (self.split == 'test') and (self.seq == -1):
                annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
                if self.pseudo_mask == 'mink_notta':
                    annotated_data_pseudo = np.fromfile(newpath.replace('data_root/SemanticKITTI', \
                        'logs/voxel/semantic_kitti/minkunet_mk34_cr10/default/test_notta').replace('velodyne', 'predictions')[:-3] + 'label',
                                                    dtype=np.uint32).reshape((-1, 1))
                    annotated_data_pseudo = annotated_data_pseudo & 0xFFFF
            else:
                if newpath.split('/sequences/')[-1][:2] in self.trainval_seqs:
                    annotated_data = np.fromfile(newpath.replace('velodyne', 'labels')[:-3] + 'label', dtype=np.uint32).reshape((-1, 1))
                    annotated_data = annotated_data & 0xFFFF

                    if self.pseudo_mask == 'mink_notta':
                        annotated_data_pseudo = np.fromfile(newpath.replace('data_root/SemanticKITTI', \
                            'logs/voxel/semantic_kitti/minkunet_mk34_cr10/default/trainval_notta').replace('velodyne', 'predictions')[:-3] + 'label',
                                                        dtype=np.uint32).reshape((-1, 1))
                        annotated_data_pseudo = annotated_data_pseudo & 0xFFFF
                    elif self.pseudo_mask == 'gt':
                        annotated_data_pseudo = annotated_data

            if np.abs(delta_idx) % step_image == 0:
                image_file = newpath.replace('velodyne', 'image_2').replace('.bin', '.png')
                raw_data_fov, cropped_image, cropped_depth_map, cropped_lidar_map, cropped_semantic_map \
                     = self.get_fov_points(copy.deepcopy(raw_data), image_file, dir_idx, np.abs(delta_idx) // step_image)

                if self.fov_dist > 0:
                  radius = np.linalg.norm(raw_data_fov[:, :2], ord=2, axis=1, keepdims=False)
                  raw_data_fov = raw_data_fov[radius <= self.fov_dist]
                raw_data_fov = self.fuse_multi_scan(raw_data_fov, pose0, pose) if delta_idx!=0 else raw_data_fov
                raw_data_fov_ms.insert(0, raw_data_fov)
                cropped_image_ms.insert(0, cropped_image)
                cropped_depth_map_ms.insert(0, cropped_depth_map)
                cropped_lidar_map_ms.insert(0, cropped_lidar_map)
                cropped_semantic_map_ms.insert(0, cropped_semantic_map)

            if (delta_idx<-multiscan) or (delta_idx >=0):
                continue

            annotated_data_mask = np.zeros(len(annotated_data), dtype=np.bool)
            for class_idx, class_step in enumerate(flexible_steps):
                if class_step == 0:
                    continue
                if np.abs(delta_idx) % class_step == 0:
                    annotated_data_mask = annotated_data_mask | (annotated_data_pseudo[:,0] == LEARNING_MAP_INV[class_idx])

            raw_data = self.fuse_multi_scan(raw_data, pose0, pose)
            raw_data_ms.append(raw_data)
            annotated_data_ms.append(annotated_data)
            annotated_data_ms_mask.append(annotated_data_mask)

        if len(raw_data_ms) == 0:
            raw_data_ms, annotated_data_ms, annotated_data_ms_mask = None, None, None
        else:
            raw_data_ms = np.concatenate(raw_data_ms, 0)
            annotated_data_ms = np.concatenate(annotated_data_ms, 0)
            annotated_data_ms_mask = np.concatenate(annotated_data_ms_mask, 0)
            annotated_data_ms = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data_ms)

        raw_data_fov_ms = np.concatenate(raw_data_fov_ms, 0)
        cropped_image_ms = np.stack(cropped_image_ms, axis=0)
        cropped_depth_map_ms = np.stack(cropped_depth_map_ms, axis=0)
        cropped_lidar_map_ms = np.stack(cropped_lidar_map_ms, axis=0)
        cropped_semantic_map_ms = np.stack(cropped_semantic_map_ms, axis=0)

        return raw_data_ms, annotated_data_ms, annotated_data_ms_mask, \
            raw_data_fov_ms, cropped_image_ms, cropped_depth_map_ms, cropped_lidar_map_ms, cropped_semantic_map_ms

    def get_fov_points(self, raw_data, image_file, dir_idx, img_batch):
        image = Image.open(image_file)
        # depth_map = np.load(image_file.replace('image_2', 'depth_map').replace('.png', '.npy'))
        # lidar_map = np.load(image_file.replace('image_2', 'lidar_map').replace('.png', '.npy'))
        depth_map = np.zeros((image.size[1], image.size[0], 1), dtype=np.float32)
        lidar_map = np.zeros((image.size[1], image.size[0], 4), dtype=np.float32)
        semantic_map = np.load(image_file.replace('image_2', 'semantic_map_dilate').replace('.png', '.npy'))
        proj_matrix = self.proj_matrix[dir_idx]

        keep_mask = raw_data[:, 0] > 0
        raw_data_front_xyz = np.concatenate([raw_data[:, :3][keep_mask], np.ones([keep_mask.sum(), 1], dtype=np.float32)], axis=1)
        raw_data_front_uvz = (proj_matrix @ raw_data_front_xyz.T).T
        raw_data_front_z = raw_data_front_uvz[:, 2]
        raw_data_front_uv = raw_data_front_uvz[:, :2] / np.expand_dims(raw_data_front_uvz[:, 2], axis=1)
        frustum_mask = self.select_points_in_frustum(raw_data_front_uv, 0, 0, *image.size)
        keep_mask[keep_mask] = frustum_mask

        raw_data_front_uv = np.fliplr(raw_data_front_uv)
        raw_data_frustum_uv = raw_data_front_uv[frustum_mask].astype(dtype=np.int)
        raw_data_frustum_z = raw_data_front_z[frustum_mask]
        
        image = np.array(image, dtype=np.float32, copy=False)

        if self.image_jitter:
            image = self.color_jitter(image)

        image[..., [0,1,2]] = image[..., [2,1,0]]
        image = image / 255.

        if self.image_flip and (np.random.rand() < self.flip_ratio):
            image = np.ascontiguousarray(np.fliplr(image))
            depth_map = np.ascontiguousarray(np.fliplr(depth_map))
            lidar_map = np.ascontiguousarray(np.fliplr(lidar_map))
            semantic_map = np.ascontiguousarray(np.fliplr(semantic_map))
            raw_data_frustum_uv[:, 1] = image.shape[1] - 1 - raw_data_frustum_uv[:, 1]

        r_max = min(self.height, image.shape[0])
        c_max = min(self.width, image.shape[1])
        cropped_image = np.zeros((self.height, self.width, 3), dtype=image.dtype)
        cropped_depth_map = np.zeros((self.height, self.width, 1), dtype=cropped_image.dtype)
        cropped_lidar_map = np.zeros((self.height, self.width, 4), dtype=cropped_image.dtype)
        cropped_semantic_map = np.zeros((self.height, self.width, 1), dtype=cropped_image.dtype)
        cropped_image[:r_max, :c_max] = image[:r_max, :c_max]
        cropped_depth_map[:r_max, :c_max] = depth_map[:r_max, :c_max]
        cropped_lidar_map[:r_max, :c_max] = lidar_map[:r_max, :c_max]
        cropped_semantic_map[:r_max, :c_max] = semantic_map[:r_max, :c_max]

        cropped_mask = (raw_data_frustum_uv[:, 0] < self.height) & (raw_data_frustum_uv[:, 1] < self.width)
        keep_mask[keep_mask.nonzero()[0][~cropped_mask]] = False
        raw_data_frustum_uv = raw_data_frustum_uv[cropped_mask]
        raw_data_frustum_uv = raw_data_frustum_uv.astype(raw_data.dtype)
        raw_data_frustum_uv[:, 0] += (self.height * img_batch)
        raw_data = np.concatenate([raw_data[keep_mask], raw_data_frustum_uv], axis=-1)

        return raw_data, cropped_image, cropped_depth_map, cropped_lidar_map, cropped_semantic_map
    
    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.proj_matrix = {}
        self.calibrations = []
        self.times = []
        self.poses = []

        for seq in range(0, 22):
            seq_folder = os.path.join(self.root_path, str(seq).zfill(2))

            # Read Calib for Painting
            calib = self.read_calib(os.path.join(seq_folder, "calib.txt"))
            proj_matrix = np.matmul(calib["P2"], calib["Tr"])
            self.proj_matrix[seq] = proj_matrix

            # Read Calib
            self.calibrations.append(self.parse_calibration(os.path.join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(os.path.join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(os.path.join(seq_folder, 'poses.txt'), self.calibrations[-1])
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out

    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def parse_calibration(self, filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def fuse_multi_scan(self, points, pose0, pose):

        # pose = poses[0][idx]

        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        # new_points = hpoints.dot(pose.T)
        new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

        new_points = new_points[:, :3]
        new_coords = new_points - pose0[:3, 3]
        # new_coords = new_coords.dot(pose0[:3, :3])
        new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
        new_coords = np.hstack((new_coords, points[:, 3:]))

        return new_coords

    def color_jitter(self, img):
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)
        
        return img

    def convert(self,
                img: np.ndarray,
                alpha: int = 1,
                beta: int = 0) -> np.ndarray:
        """Multiple with alpha and add beat with clip.

        Args:
            img (np.ndarray): The input image.
            alpha (int): Image weights, change the contrast/saturation
                of the image. Default: 1
            beta (int): Image bias, change the brightness of the
                image. Default: 0

        Returns:
            np.ndarray: The transformed image.
        """

        # img = img.astype(np.float32) * alpha + beta
        img = img * alpha + beta
        img = np.clip(img, 0, 255)
        return img
        # return img.astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:
        """Brightness distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after brightness change.
        """

        if np.random.randint(2):
            return self.convert(
                img,
                beta=np.random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        """Contrast distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after contrast change.
        """

        if np.random.randint(2):
            return self.convert(
                img,
                alpha=np.random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:
        """Saturation distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after saturation change.
        """

        if np.random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=np.random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:
        """Hue distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after hue change.
        """

        if np.random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      np.random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img