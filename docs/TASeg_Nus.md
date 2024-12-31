## Temporal LiDAR Aggregation and Distillation
#### Train a single-frame model
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh dist_train.sh 4  \
--cfg_file tools/cfgs/voxel/nuscenes/minkunet_mk34_cr10.yaml
```

#### Prepare history predictions for the multi-frame model with the single-frame model
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh dist_train.sh 4  \
--cfg_file tools/cfgs/voxel/nuscenes/minkunet_mk34_cr10.yaml --workers 8 \
--eval --tta --seq -2 --votes_min 0  --votes_max 1 --batch_size 4 \
--save_path /YourHome/PCSeg/logs/voxel/nuscenes/minkunet_mk34_cr10/default/results/lidarseg/trainval_sweep_notta \
--ckp /YourHome/PCSeg/logs/voxel/nuscenes/minkunet_mk34_cr10/default/ckp/checkpoint_epoch_48.pth \
--set DATA.DATASET nuscenes_sweep
```

#### Train a multi-frame model with FSA
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh dist_train.sh 4  \
--cfg_file tools/cfgs/voxel/nuscenes/minkunet_mk34_cr10_fsa.yaml
```

## Temporal Image Feature Aggregation
#### Prepare 2D Semantic Segmentation Labels
```
```

#### Perform temporal multi-modal fusion
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh dist_train.sh 4  \
--cfg_file tools/cfgs/voxel/nuscenes/minkunet_mk34_cr10_fsa_tiaf.yaml --fix_part_param \
--pretrained_model /YourHome/PCSeg/logs/voxel/nuscenes/minkunet_mk34_cr10_fsa/default/ckp/checkpoint_epoch_36.pth
```