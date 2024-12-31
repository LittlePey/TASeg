# Getting Started

### Semantic KITTI

For example, if you want to train the following models with 2 GPUs:

```
CUDA_VISIBLE_DEVICES=0,1 sh dist_train.sh 2 --cfg_file tools/cfgs/voxel/minkunet_mk34_cr10.yaml           

CUDA_VISIBLE_DEVICES=2,3 sh dist_train.sh 2 --cfg_file tools/cfgs/voxel/cylinder_cy480_cr10.yaml

CUDA_VISIBLE_DEVICES=4,5 sh dist_train.sh 2 --cfg_file tools/cfgs/fusion/spvcnn_mk18_cr10.yaml

CUDA_VISIBLE_DEVICES=6,7 sh dist_train.sh 2 --cfg_file tools/cfgs/fusion/rpvnet_mk18_cr10.yaml
```
