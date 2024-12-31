### 1. Flexible Step Aggregation
#### Train a single-frame model
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh dist_train.sh 4  \
--cfg_file tools/cfgs/voxel/semantic_kitti/minkunet_mk34_cr10.yaml
```

#### Prepare history predictions for the multi-frame model with the single-frame model
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh dist_train.sh 8 \
--cfg_file tools/cfgs/voxel/semantic_kitti/minkunet_mk34_cr10.yaml \
--eval --tta --seq -2 --votes_min 0  --votes_max 1 --batch_size 8 \
--save_path /YourHome/PCSeg/logs/voxel/semantic_kitti/minkunet_mk34_cr10/default/trainval_notta \
--ckp /YourHome/PCSeg/logs/voxel/semantic_kitti/minkunet_mk34_cr10/default/ckp/checkpoint_epoch_36.pth
python tta_remap.py -p /YourHome/PCSeg/logs/voxel/semantic_kitti/minkunet_mk34_cr10/default/trainval_notta -s trainval --inverse
```

#### Train a multi-frame model with FSA
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh dist_train.sh 4  \
--cfg_file tools/cfgs/voxel/semantic_kitti/minkunet_mk34_cr10_fsa.yaml
```

### 2. Mask Distillation
#### Train a multi-frame model with GT mask
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh dist_train.sh 4  \
--cfg_file tools/cfgs/voxel/semantic_kitti/minkunet_mk34_cr10_fsa.yaml --extra_tag teacher \
--set DATA.PSEUDO_MASK gt
```

#### Perform Mask Distillation
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh dist_train.sh 4  \
--cfg_file tools/cfgs/voxel/semantic_kitti/minkunet_mk34_cr10_fsa_kd.yaml --fix_part_param \
--pretrained_model /YourHome/PCSeg/logs/voxel/semantic_kitti/minkunet_mk34_cr10_fsa/teacher/ckp/checkpoint_epoch_12.pth
```

## 3.Temporal Image Feature Aggregation
#### Prepare 2D Semantic Segmentation Labels
```
```

#### Perform temporal multi-modal fusion
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh dist_train.sh 4  \
--cfg_file tools/cfgs/voxel/semantic_kitti/minkunet_mk34_cr10_fsa_tiaf.yaml --fix_part_param \
--pretrained_model /YourHome/PCSeg/logs/voxel/semantic_kitti/minkunet_mk34_cr10_fsa_kd/default/ckp/checkpoint_epoch_12.pth

```

## 4.Static-Moving Switch Augmentation
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh dist_train.sh 4  \
--cfg_file tools/cfgs/voxel/semantic_kitti_ms/minkunet_mk34_cr10_smsa.yaml
```
