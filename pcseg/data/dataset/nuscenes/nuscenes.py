import os
import numpy as np
from torch.utils import data
from .LaserMix_nuscenes import lasermix_aug, lasermix_aug_
from .PolarMix_nuscenes import polarmix
import random
import pickle
import yaml
import pdb

# used for polarmix
instance_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]

class NuscenesDataset(data.Dataset):
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
        self.data_path_ceph = data_cfgs.get('DATA_PATH_CEPH', None) # "cluster5:s3://wxp-DataSets/nuscenes/"

        if self.training:
            self.split = 'train'
        else:
            self.split = 'val'
        if self.tta:
            self.split = 'test'

        from nuscenes import NuScenes
        if self.split == 'test':
            self.nusc = NuScenes(version='v1.0-test', dataroot=root_path, verbose=True)
        else:
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=root_path, verbose=True)

        if self.seq == -2:
            with open(os.path.join(self.root_path, data_cfgs.INFO_PATH['train']), 'rb') as f:
                data_train = pickle.load(f)
            with open(os.path.join(self.root_path, data_cfgs.INFO_PATH['val']), 'rb') as f:
                data_val = pickle.load(f)
            self.nusc_infos = data_train['infos'] + data_val['infos']
        else:
            with open(os.path.join(self.root_path, data_cfgs.INFO_PATH[self.split]), 'rb') as f:
                data = pickle.load(f)
            self.nusc_infos = data['infos']

        self.nusc_infos_another = self.nusc_infos.copy()
        random.shuffle(self.nusc_infos_another)
        print(f'The total sample is {len(self.nusc_infos)}')

        with open('./pcseg/data/dataset/nuscenes/nuscenes.yaml', 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self._sample_idx = np.arange(len(self.nusc_infos))
        self.samples_per_epoch = self.data_cfgs.get('SAMPLES_PER_EPOCH', -1)
        if self.samples_per_epoch == -1 or not self.training:
            self.samples_per_epoch = len(self.nusc_infos)

        if self.training:
            self.resample()
        else:
            self.sample_idx = self._sample_idx

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

        prob = np.random.choice(2, 1)
        if self.augment == 'GlobalAugment_LP' or self.augment == 'GlobalAugment_L' or self.augment == 'GlobalAugment_P':
            if self.split == 'train' and (self.augment == 'GlobalAugment_LP' or self.augment == 'GlobalAugment_L') and prob == 1:
                info1 = self.nusc_infos_another[index]
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
                raw_data, annotated_data = lasermix_aug(
                    raw_data,
                    annotated_data,
                    raw_data1,
                    annotated_data1,
                )
            
            elif self.split == 'train' and (self.augment == 'GlobalAugment_LP' or self.augment == 'GlobalAugment_P') and prob == 0:
                info1 = self.nusc_infos_another[index]
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
                alpha = (np.random.random() - 1) * np.pi
                beta = alpha + np.pi
                annotated_data1 = annotated_data1.reshape(-1)
                annotated_data = annotated_data.reshape(-1)
                raw_data, annotated_data = polarmix(
                    raw_data, annotated_data, raw_data1, annotated_data1,
                    alpha=alpha, beta=beta,
                    instance_classes=instance_classes, Omega=Omega
                )
                annotated_data = annotated_data.reshape(-1, 1)
        
        pc_data = {
            'xyzret': raw_data,
            'labels': annotated_data.astype(np.uint8),
            'path': lidarseg_labels_filename,
        }

        return pc_data

    @staticmethod
    def collate_batch(batch_list):
        raise NotImplementedError

