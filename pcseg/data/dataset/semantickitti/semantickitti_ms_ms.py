import os
import numpy as np
from torch.utils import data
from .semantickitti_utils import LEARNING_MAP as LEARNING_MAP_SS
from .semantickitti_utils import LEARNING_MAP_INV as LEARNING_MAP_INV_SS
from .semantickitti_utils_ms_ms import LEARNING_MAP, LEARNING_MAP_INV
from .LaserMix_semantickitti import lasermix_aug
from .PolarMix_semantickitti import polarmix
import random
from typing import List, Tuple, Union
from itertools import repeat
from pcseg.data.dataset.ceph import PetrelBackend

# used for polarmix
instance_classes = [1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23, 24, 25]
Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]

def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


class SemantickittiMsMsDataset(data.Dataset):
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
        self.augment = data_cfgs.AUGMENT
        self.if_scribble = if_scribble

        self.maug_prob = data_cfgs.get('MAUG_PROB')
        self.shift_x_range = data_cfgs.get('SHIFT_X_RANGE')
        self.shift_y_range = data_cfgs.get('SHIFT_Y_RANGE')
        
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

        self.multiscan      = self.data_cfgs.MULTISCAN
        self.only_history    = self.data_cfgs.ONLY_HISTORY
        self.pseudo_mask   = self.data_cfgs.PSEUDO_MASK
        self.flexible_steps = self.data_cfgs.FLEXIBLE_STEPS
        self.step = 1

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
                    instance_id = annotated_data.copy().reshape(-1)
            
            annotated_data = annotated_data & 0xFFFF
            annotated_data_raw = annotated_data.copy()
            annotated_data = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data)

        if int(self.annos[index][-10:-4]) - 1 >=0:
            raw_data_ms, annotated_data_ms,  annotated_data_ms_mask, instance_id_ms, annotated_data_ms_raw, time_flag_ms \
                        = self.multiscan_fuse(self.annos, index, self.multiscan, self.flexible_steps)
            if (self.split == 'train') and (len(instance_id_ms) > 0) and  (((annotated_data_raw == 18) | (annotated_data_raw == 20)).sum() > 0):
                raw_data, annotated_data_raw, raw_data_ms, annotated_data_ms_raw \
                    = self.static2moving(raw_data, annotated_data_raw, instance_id, \
                                         raw_data_ms, annotated_data_ms_raw, instance_id_ms, time_flag_ms, index)
                annotated_data = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data_raw)
                annotated_data_ms = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data_ms_raw)
            if (self.split == 'train') and (len(instance_id_ms) > 0) and  (((annotated_data_raw == 253) | (annotated_data_raw == 255)).sum() > 0):
                raw_data, annotated_data_raw, raw_data_ms, annotated_data_ms_raw \
                    = self.moving2static(raw_data, annotated_data_raw, instance_id, \
                                         raw_data_ms, annotated_data_ms_raw, instance_id_ms, time_flag_ms, index)
                annotated_data = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data_raw)
                annotated_data_ms = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data_ms_raw)

            raw_data_ms = np.concatenate([raw_data, raw_data_ms[annotated_data_ms_mask]])
            raw_data_ms = self.append_time_flag(raw_data, raw_data_ms)
            annotated_data_ms = np.concatenate([annotated_data, annotated_data_ms[annotated_data_ms_mask]])
        else:
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
                        instance_id1 = annotated_data1.copy().reshape(-1)
                
                annotated_data1 = annotated_data1 & 0xFFFF
                annotated_data1_raw = annotated_data1.copy()
                annotated_data1 = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data1)
                assert len(annotated_data1) == len(raw_data1)

                if int(self.annos_another[index][-10:-4]) - 1 >=0:
                    raw_data1_ms, annotated_data1_ms,  annotated_data1_ms_mask, instance_id1_ms, annotated_data1_ms_raw, time_flag1_ms \
                                = self.multiscan_fuse(self.annos_another, index, self.multiscan, self.flexible_steps)
                    if (len(instance_id1_ms) > 0) and  (((annotated_data1_raw == 18) | (annotated_data1_raw == 20)).sum() > 0):
                        raw_data1, annotated_data1_raw, raw_data1_ms, annotated_data1_ms_raw \
                            = self.static2moving(raw_data1, annotated_data1_raw, instance_id1, \
                                                 raw_data1_ms, annotated_data1_ms_raw, instance_id1_ms, time_flag1_ms)
                        annotated_data1 = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data1_raw)
                        annotated_data1_ms = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data1_ms_raw)
                    if (len(instance_id1_ms) > 0) and  (((annotated_data1_raw == 253) | (annotated_data1_raw == 255)).sum() > 0):
                        raw_data1, annotated_data1_raw, raw_data1_ms, annotated_data1_ms_raw \
                            = self.moving2static(raw_data1, annotated_data1_raw, instance_id1, \
                                                 raw_data1_ms, annotated_data1_ms_raw, instance_id1_ms, time_flag1_ms)
                        annotated_data1 = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data1_raw)
                        annotated_data1_ms = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data1_ms_raw)

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
                        instance_id1 = annotated_data1.copy().reshape(-1)
                
                annotated_data1 = annotated_data1 & 0xFFFF
                annotated_data1_raw = annotated_data1.copy()
                annotated_data1 = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data1)
                assert len(annotated_data1) == len(raw_data1)

                if int(self.annos_another[index][-10:-4]) - 1 >=0:
                    raw_data1_ms, annotated_data1_ms,  annotated_data1_ms_mask, instance_id1_ms, annotated_data1_ms_raw, time_flag1_ms \
                                = self.multiscan_fuse(self.annos_another, index, self.multiscan, self.flexible_steps)
                    if (len(instance_id1_ms) > 0) and  (((annotated_data1_raw == 18) | (annotated_data1_raw == 20)).sum() > 0):
                        raw_data1, annotated_data1_raw, raw_data1_ms, annotated_data1_ms_raw \
                            = self.static2moving(raw_data1, annotated_data1_raw, instance_id1, \
                                                 raw_data1_ms, annotated_data1_ms_raw, instance_id1_ms, time_flag1_ms)
                        annotated_data1 = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data1_raw)
                        annotated_data1_ms = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data1_ms_raw)
                    if (len(instance_id1_ms) > 0) and  (((annotated_data1_raw == 253) | (annotated_data1_raw == 255)).sum() > 0):
                        raw_data1, annotated_data1_raw, raw_data1_ms, annotated_data1_ms_raw \
                            = self.moving2static(raw_data1, annotated_data1_raw, instance_id1, \
                                                 raw_data1_ms, annotated_data1_ms_raw, instance_id1_ms, time_flag1_ms)
                        annotated_data1 = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data1_raw)
                        annotated_data1_ms = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data1_ms_raw)

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
        pc_data = {
            'xyzret': raw_data,
            'xyzret_ms': raw_data_ms,
            'labels': annotated_data.astype(np.uint8),
            'labels_ms': annotated_data_ms.astype(np.uint8),
            'path': self.annos[index],
        }

        return pc_data

    def static2moving(self, raw_data, annotated_data_raw, instance_id, raw_data_ms, annotated_data_ms_raw, instance_id_ms, time_flag_ms, index=-1):
        instance_label = instance_id[((annotated_data_raw==18) | (annotated_data_raw==20)).reshape(-1)].squeeze()
        unique_label = np.unique(instance_label)

        for inst in unique_label:
            maug_prob = np.random.choice(self.maug_prob, 1)
            if maug_prob != 1:
                continue

            instance_mask = instance_id == inst
            instance_mask_ms = instance_id_ms == inst
            instance_pc_ms = raw_data_ms[instance_mask_ms]

            if instance_mask_ms.sum()==0:
                continue
            if instance_pc_ms[:, 0].max() - instance_pc_ms[:, 0].min() > instance_pc_ms[:, 1].max() - instance_pc_ms[:, 1].min():
                center_y = instance_pc_ms[:, 1].mean()
                if center_y > 4:
                    center_shift = 2 + np.random.rand() * 3
                    raw_data_ms[instance_mask_ms, 1] -= center_shift
                    raw_data[instance_mask, 1] -= center_shift
                elif center_y < -2:
                    center_shift = 2 + np.random.rand() * 3
                    raw_data_ms[instance_mask_ms, 1] += center_shift
                    raw_data[instance_mask, 1] += center_shift

                shift_x = np.random.rand() * self.shift_x_range + 0.5
                for delta_idx in range(-self.multiscan, self.multiscan + self.step, self.step):
                    if self.only_history and delta_idx > 0:
                        continue
                    delta_mask = time_flag_ms == delta_idx
                    raw_data_ms[instance_mask_ms & delta_mask, 0] += (delta_idx / self.step * shift_x)

            else:
                shift_y = np.random.rand() * self.shift_y_range + 0.5
                for delta_idx in range(-self.multiscan, self.multiscan + self.step, self.step):
                    if self.only_history and delta_idx > 0:
                        continue
                    delta_mask = time_flag_ms == delta_idx
                    raw_data_ms[instance_mask_ms & delta_mask, 1] += (delta_idx / self.step * shift_y)

            annotated_data_raw[instance_mask & (annotated_data_raw==18).reshape(-1)] = 258
            annotated_data_raw[instance_mask & (annotated_data_raw==20).reshape(-1)] = 259
            annotated_data_ms_raw[instance_mask_ms & (annotated_data_ms_raw==18).reshape(-1)] = 258
            annotated_data_ms_raw[instance_mask_ms & (annotated_data_ms_raw==20).reshape(-1)] = 259

        return raw_data, annotated_data_raw, raw_data_ms, annotated_data_ms_raw

    def moving2static(self, raw_data, annotated_data_raw, instance_id, raw_data_ms, annotated_data_ms_raw, instance_id_ms, time_flag_ms, index=-1):
        instance_label = instance_id[((annotated_data_raw==253) | (annotated_data_raw==255)).reshape(-1)].squeeze()
        unique_label = np.unique(instance_label)

        for inst in unique_label:
            maug_prob = np.random.choice(self.maug_prob, 1)
            if maug_prob != 1:
                continue

            instance_mask = instance_id == inst
            instance_mask_ms = instance_id_ms == inst
            if (instance_mask.sum() < 20) or (instance_mask_ms.sum()==0):
                continue

            instance_pc_cur = raw_data[instance_mask]
            instance_pc_pre = raw_data_ms[instance_mask_ms & (time_flag_ms == -self.step)]
            shift_x = (instance_pc_pre[:, 0].mean() - instance_pc_cur[:, 0].mean())
            shift_y = (instance_pc_pre[:, 1].mean() - instance_pc_cur[:, 1].mean())

            for delta_idx in range(-self.multiscan, self.multiscan + self.step, self.step):
                if self.only_history and delta_idx > 0:
                    continue
                delta_mask = time_flag_ms == delta_idx
                raw_data_ms[instance_mask_ms & delta_mask, 0] += (delta_idx / self.step * shift_x)
                raw_data_ms[instance_mask_ms & delta_mask, 1] += (delta_idx / self.step * shift_y)

            annotated_data_raw[instance_mask & (annotated_data_raw==253).reshape(-1)] = 31
            annotated_data_ms_raw[instance_mask_ms & (annotated_data_ms_raw==253).reshape(-1)] = 31
            annotated_data_raw[instance_mask & (annotated_data_raw==255).reshape(-1)] = 32
            annotated_data_ms_raw[instance_mask_ms & (annotated_data_ms_raw==255).reshape(-1)] = 32

        return raw_data, annotated_data_raw, raw_data_ms, annotated_data_ms_raw

    def append_time_flag(self, raw_data, raw_data_ms):
        time_flag = np.zeros((len(raw_data_ms), 1), dtype=raw_data_ms.dtype)
        time_flag[:len(raw_data), 0] = 1
        raw_data_ms = np.concatenate([raw_data_ms[:, :4], time_flag, raw_data_ms[:, 4:]], axis=1)
        return raw_data_ms

    @staticmethod
    def collate_batch(batch_list):
        raise NotImplementedError

    def multiscan_fuse(self, annos, index, multiscan, flexible_steps):
        raw_data_ms = []
        annotated_data_ms = []
        annotated_data_ms_mask = []
        instance_id_ms = []
        time_flag_ms = []

        number_idx = int(annos[index][-10:-4])
        dir_idx = int(annos[index][-22:-20])
        pose0 = self.poses[dir_idx][number_idx]
        for delta_idx in range(-multiscan, multiscan):
            if delta_idx == 0:
                continue
            if self.only_history and delta_idx > 0:
                continue
            try:
                pose = self.poses[dir_idx][number_idx + delta_idx]
                newpath = annos[index][:-10] + str(number_idx + delta_idx).zfill(6) + annos[index][-4:]
                raw_data = np.fromfile(newpath, dtype=np.float32).reshape((-1, 4))
            except:
                continue

            instance_id = None
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
                    instance_id = annotated_data.copy().reshape(-1)
                    annotated_data = annotated_data & 0xFFFF

                    if self.pseudo_mask == 'mink_notta':
                        annotated_data_pseudo = np.fromfile(newpath.replace('data_root/SemanticKITTI', \
                            'logs/voxel/semantic_kitti/minkunet_mk34_cr10/default/trainval_notta').replace('velodyne', 'predictions')[:-3] + 'label',
                                                        dtype=np.uint32).reshape((-1, 1))
                        annotated_data_pseudo = annotated_data_pseudo & 0xFFFF
                    elif self.pseudo_mask == 'gt':
                        annotated_data_pseudo = annotated_data
            
            annotated_data_mask = np.zeros(len(annotated_data), dtype=np.bool)
            for class_idx, class_step in enumerate(flexible_steps):
                if class_step == 0:
                    continue
                if np.abs(delta_idx) % class_step == 0:
                    annotated_data_mask = annotated_data_mask | (annotated_data_pseudo[:,0] == LEARNING_MAP_INV[class_idx])

            raw_data = self.fuse_multi_scan(raw_data, pose0, pose)
            time_flag = np.ones((len(raw_data)), dtype=raw_data.dtype) * delta_idx
            raw_data_ms.append(raw_data)
            annotated_data_ms.append(annotated_data)
            annotated_data_ms_mask.append(annotated_data_mask)
            if instance_id is not None:
                instance_id_ms.append(instance_id)
            time_flag_ms .append(time_flag)

        raw_data_ms = np.concatenate(raw_data_ms, 0)
        annotated_data_ms = np.concatenate(annotated_data_ms, 0)
        annotated_data_ms_mask = np.concatenate(annotated_data_ms_mask, 0)
        if len(instance_id_ms) > 0:
            instance_id_ms = np.concatenate(instance_id_ms, 0)
        time_flag_ms = np.concatenate(time_flag_ms, 0)

        annotated_data_ms_raw = annotated_data_ms.copy()
        annotated_data_ms = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data_ms)
        return raw_data_ms, annotated_data_ms, annotated_data_ms_mask, instance_id_ms, annotated_data_ms_raw, time_flag_ms

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        self.times = []
        self.poses = []

        for seq in range(0, 22):
            seq_folder = os.path.join(self.root_path, str(seq).zfill(2))

            # Read Calib
            self.calibrations.append(self.parse_calibration(os.path.join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(os.path.join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(os.path.join(seq_folder, 'poses.txt'), self.calibrations[-1])
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

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

