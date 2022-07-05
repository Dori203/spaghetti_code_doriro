from __future__ import annotations
import os
import pickle5 as pickle
import constants as const
from custom_types import *


class Options:

    @property
    def num_levels(self) -> int:
        return len(self.hierarchical)

    def load(self) -> object:
        device = self.device
        if os.path.isfile(self.save_path):
            print(f'loading opitons from {self.save_path}')
            with open(self.save_path, 'rb') as f:
                options = pickle.load(f)
            options = backward_compatibility(options)
            options.device = device
            return options
        return self

    def save(self):
        if os.path.isdir(self.cp_folder):
            # self.already_saved = True
            with open(self.save_path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @property
    def info(self) -> str:
        return f'{self.model_name}_{self.tag}'

    @property
    def cp_folder(self):
        return f'{const.CHECKPOINTS_ROOT}{self.info}'

    @property
    def flask_public_folder(self):
        return f'{const.FLASK_PUBLIC}'

    @property
    def archive_image_folder(self):
        return f'{const.ARCHIVE_IMAGES}'

    @property
    def archive_json_folder(self):
        return f'{const.ARCHIVE_JSON}'

    @property
    def save_path(self):
        return f'{const.CHECKPOINTS_ROOT}{self.info}/options.pkl'

    def fill_args(self, args):
        for arg in args:
            if hasattr(self, arg):
                setattr(self, arg, args[arg])

    def __init__(self, **kwargs):
        self.device = CUDA(0)
        self.tag = 'chairs_sin_subset'
        self.dataset_name = 'shapenet_chairs_wm_sphere_sym_train'
        self.epochs = 2700
        self.model_name = 'occ_gmm'
        self.dim_z = 256
        self.pos_dim = 256 - 3
        self.dim_h = 512
        self.dim_zh = 512
        self.num_gaussians = 16
        self.min_split = 4
        self.max_split = 12
        self.gmm_weight = 1
        self.num_layers = 4
        self.num_heads = 4
        self.batch_size = 10
        self.num_samples = 2000
        self.dataset_size = -1
        self.variational = False
        self.symmetric = (True, False, False)
        self.symmetric_loss = (False, False, False)
        self.data_symmetric = (True, False, False)
        self.variational_gamma = 1.e-1
        self.reset_ds_every = 100
        self.plot_every = 100
        self.lr_decay = .9
        self.lr_decay_every = 500
        self.warm_up = 2000
        self.temperature_decay = .99
        self.loss_func = [LossType.CROSS, LossType.HINGE, LossType.IN_OUT][2]
        self.decomposition_network = 'transformer'
        self.head_type = "deep_sdf"
        self.head_sdf_size = 5
        self.reg_weight = 1e-4
        self.num_layers_head = 6
        self.num_heads_head = 8
        self.disentanglement = True
        self.use_encoder = True
        self.disentanglement_weight = 1
        self.augmentation_rotation = 0.3
        self.augmentation_scale = .3
        self.augmentation_translation = .3
        self.as_tait_bryan = False
        self.hierarchical = ()
        self.mask_head_by_gmm = 0
        self.pos_encoding_type = 'sin'
        self.subset = 100
        self.fill_args(kwargs)

# class ProjectionOptions(Options):
#
#     def __init__(self):
#         super(ProjectionOptions, self).__init__()
#         self.epochs = 2000

class OptionsSingle(Options):

    def __init__(self, **kwargs):
        super(OptionsSingle, self).__init__(**kwargs)
        self.tag = 'single_wolf_prune'
        self.dataset_name = 'MalteseFalconSolid'
        self.dim_z = 64
        self.pos_dim = 64 - 3
        self.dim_h = 64
        self.dim_zh = 64
        self.num_gaussians = 12
        self.gmm_weight = 1
        self.batch_size = 18
        self.num_samples = 3000
        self.dataset_size = 1
        self.symmetric = (False, False, False)
        self.head_type = "deep_sdf"
        self.head_sdf_size = 3
        self.reg_weight = 1e-4
        self.num_layers_head = 4
        self.num_heads_head = 4
        self.disentanglement = True
        self.disentanglement_weight = 1
        self.augmentation_rotation = .5
        self.augmentation_scale = .3
        self.augmentation_translation = .3
        self.prune_every = 200
        self.fill_args(kwargs)


class OptionsDiscriminator(Options):

    def __init__(self, **kwargs):
        super(OptionsDiscriminator, self).__init__(**kwargs)
        self.model_name = 'discriminator'
        self.discriminator_num_layers = 4
        self.discriminator_dim = 8


def backward_compatibility(opt: Options) -> Options:
    defaults = {'as_tait_bryan': False, 'head_type': "deep_sdf", "hierarchical": (),
                'decomposition_network': 'transformer',
                'mask_head_by_gmm': 0, 'use_encoder': True, 'data_symmetric': opt.symmetric,
                'symmetric_loss': (False, False, False), 'pos_encoding_type': 'sin',
                'subset': -1}
    for key, item in defaults.items():
        if not hasattr(opt, key):
            setattr(opt, key, item)
    return opt
