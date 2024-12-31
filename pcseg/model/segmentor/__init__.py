# raw point
# ...

# range view
from .range.rangenet.model.semantic.rangenet import RangeNet
from .range.salsanext.model.semantic.salsanext import SalsaNext
from .range.fidnet.model.semantic.fidnet import FIDNet
from .range.cenet.model.semantic.cenet import CENet

# bird's eye view
# ...

# voxel
from .voxel.cylinder3d import Cylinder_TS
from .voxel.cylinder3d.cylinder_ts import Cylinder_TS
from .voxel.minkunet.minkunet import MinkUNet

from .voxel.minkunet.minkunet_ms import MinkUNetMs
from .voxel.minkunet.minkunet_ms_kd import MinkUNetMsKd
from .voxel.minkunet.minkunet_ms_mm import MinkUNetMsMm
from .voxel.minkunet.minkunet_ms_mm_nus import MinkUNetMsMmNus

# multi-view fusion
from .fusion.spvcnn.spvcnn import SPVCNN #, MinkUNet
from .fusion.rpvnet.rpvnet import RPVNet



__all__ = {
    # raw point
    # ...

    # range view
    'RangeNet++': RangeNet,
    'SalsaNext': SalsaNext,
    'FIDNet': FIDNet,
    'CENet': CENet,

    # bird's eye view
    # ...

    # voxel
    'Cylinder_TS': Cylinder_TS,
    'MinkUNet': MinkUNet,
    'MinkUNetMs': MinkUNetMs,
    'MinkUNetMsKd': MinkUNetMsKd,
    'MinkUNetMsMm': MinkUNetMsMm,
    'MinkUNetMsMmNus': MinkUNetMsMmNus,

    # multi-view fusion
    'SPVCNN': SPVCNN,
    'RPVNet': RPVNet,
}


def build_segmentor(model_cfgs, num_class):
    model = eval(model_cfgs.NAME)(
        model_cfgs=model_cfgs,
        num_class=num_class,
    )

    return model
