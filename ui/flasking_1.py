from flask import Flask, jsonify, request

# ------------------------------------- occ inference imports

import sys

sys.path.insert(0, 'C:/Users/dori2/Desktop/Bezalel/Year 5/pgmr/spaghetti_code/spaghetti_code')
import utils.rotation_utils
from custom_types import *
import constants
from data_loaders import mesh_datasets
from options import Options
from utils import train_utils, mcubes_meshing, files_utils, mesh_utils
from models.occ_gmm import OccGen
from models import models_utils
from flask import request
from flask_cors import CORS
import json
import trimesh
import copy
from PIL import Image
from pyglet import gl
import open3d

# ------------------------------------- occ inference classes
restart_counter = 0

class Inference:

    def split_shape(self, mu: T) -> T:
        b, g, c = mu.shape
        # rotation_mat = mesh_utils.get_random_rotation(b).to(self.device)
        # mu_z: T = torch.einsum('bad,bgd->bga', rotation_mat, mu)[:, :, -1]
        mask = []
        for i in range(b):
            axis = torch.randint(low=0, high=c, size=(1,)).item()
            random_down_top_order = mu[i, :, axis].argsort(dim=-1, descending=torch.randint(low=0, high=2,
                                                                                            size=(1,)).item() == 0)
            split_index = g // 4 + torch.randint(g // 2, size=(1,), device=self.device)
            mask.append(random_down_top_order.lt(split_index))  # True- the gaussians we drop
        return torch.stack(mask, dim=0)

    def mix_z(self, gmms, zh):
        with torch.no_grad():
            mu = gmms[0].squeeze(1)[:self.opt.batch_size // 2, :]
            mask = self.split_shape(mu).float().unsqueeze(-1)
            zh_fake = zh[:self.opt.batch_size // 2] * mask + (1 - mask) * zh[self.opt.batch_size // 2:]
        return zh_fake

    def get_occ_fun(self, z: T, gmm: Optional[TS] = None):

        def forward(x: T) -> T:
            nonlocal z
            x = x.unsqueeze(0)
            out = self.model.occ_head(x, z, gmm)[0, :]
            if self.opt.loss_func == LossType.CROSS:
                out = out.softmax(-1)
                out = -1 * out[:, 0] + out[:, 2]
                # out = out.argmax(-1).float() - 1
            elif self.opt.loss_func == LossType.IN_OUT:
                out = 2 * out.sigmoid_() - 1
            else:
                out.clamp_(-.2, .2)
            # torch.nan_to_num_(out, nan=1)
            return out

        if z.dim() == 2:
            z = z.unsqueeze(0)
        return forward

    def get_mesh(self, z: T, res: int, gmm: Optional[TS], get_time=False) -> Optional[T_Mesh]:

        with torch.no_grad():
            # samples = torch.rand(50000, 3, device=self.device)
            # out = forward(samples)
            # mask = out.lt(0)
            # mesh = samples[mask]
            if get_time:
                time_a = self.meshing.occ_meshing(self.get_occ_fun(z, gmm), res=res, get_time=get_time)

                time_b = sdf_mesh.create_mesh_old(self.get_occ_fun(z, gmm), device=self.opt.device,
                                                  scale=self.plot_scale, verbose=False,
                                                  res=res, get_time=get_time)

                return time_a, time_b
            else:
                mesh = self.meshing.occ_meshing(self.get_occ_fun(z, gmm), res=res)
                return mesh

    def plot_occ(self, z: Union[T, TS], gmms: Optional[List[TS]], prefix: str, res=200, verbose=False,
                 use_item_id: bool = False, fixed_items: Optional[Union[T, List[str]]] = None):
        gmms = gmms[-1]
        if type(fixed_items) is T:
            fixed_items = [f'{fixed_items[item].item():02d}' for item in fixed_items]
        for i in range(len(z)):
            gmm_ = [gmms[j][i].unsqueeze(0) for j in range(len(gmms))]
            mesh = self.get_mesh(z[i], res, [gmm_])
            if use_item_id and fixed_items is not None:
                name = f'_{fixed_items[i]}'
            elif len(z) == 1:
                name = ''
            else:
                name = f'_{i:02d}'
            if mesh is not None:
                # probs = gm_utils.gm_loglikelihood_loss((gmms,), mesh[0].unsqueeze(0), raw=True)[0][0]
                # colors = palette[probs.argmax(0)]
                path = files_utils.export_mesh(mesh, f'{self.opt.cp_folder}/occ_{prefix}/{prefix}{name}')
                # files_utils.save_np(mesh[1], f'{self.opt.cp_folder}/vox/{prefix}{name}') #TODO Doriro - good export for interpolation?
                if gmms is not None:
                    files_utils.export_gmm(gmms, i, f'{self.opt.cp_folder}/gmms_{prefix}/{prefix}{name}')
            if verbose:
                print(f'done {i + 1:d}/{len(z):d}')
        return path

    def disentanglement_plot(self, item_a: int, item_b: int, z_in_a, z_in_b, a_inclusive: bool = True,
                             b_inclusive: bool = True):

        def merge_z(z_):
            nonlocal z_in_a, z_in_b, a_inclusive, b_inclusive
            masks = []
            for inds, inclusive in zip((z_in_a, z_in_b), (a_inclusive, b_inclusive)):
                mask_ = torch.zeros(z_.shape[1], dtype=torch.bool)
                mask_[torch.tensor(inds, dtype=torch.long)] = True
                if not inclusive:
                    mask_ = ~mask_
                masks.append(mask_.to(self.device))
            z_a = z_[0][masks[0]]
            z_b = z_[0][~masks[0]]
            if item_b >= 0:
                z_a = torch.cat((z_a, z_[1][masks[1]]), dim=0)
                z_b = torch.cat((z_b, z_[1][~masks[1]]), dim=0)
            return z_a, z_b

        bottom = False
        if item_a < 0 and item_b < 0:
            return
        elif item_a < 0:
            item_a, item_b, z_in_a, z_in_b = item_b, item_a, z_in_b, z_in_a
        suffix = '' if item_b < 0 else f'_{item_b}'
        z_in_a, z_in_b = list(set(z_in_a)), list(set(z_in_b))

        with torch.no_grad():
            if item_b < 0:
                items = torch.tensor([item_a], dtype=torch.int64, device=self.device)
            else:
                items = torch.tensor([item_a, item_b], dtype=torch.int64, device=self.device)
            z_items, z_init, zh, gmms = self.model.get_disentanglement(items)
            if bottom:
                z_in = [z_.unsqueeze(0) for z_ in merge_z(z_init)]
                z_in = [self.model.sdformer.forward_split(self.model.sdformer.forward_upper(z_))[0][0] for z_ in z_in]
            else:
                z_in = merge_z(zh)
            # self.plot_sdfs(zh, gmms, f'trial_a', verbose=True)
            # z_target = replace_half(zh, gmms)
            # z_in = torch.stack((zh[0], z_target), dim=0)

            # z, gmms = self.model_a.interpolate_higher(z_in, num_between=self.fixed_items.shape[0])

            self.plot_sdfs(z_in, None, f'dist_{item_a}{suffix}', verbose=True)

    def compose(self, items: List[int], parts: List[List[int]], inclusive: List[bool]):

        def merge_z() -> T:
            nonlocal inclusive, z_parts, zh
            z_ = []
            for i, (inds, inclusive) in enumerate(zip(z_parts, inclusive)):
                mask_ = torch.zeros(zh.shape[1], dtype=torch.bool)
                mask_[torch.tensor(inds, dtype=torch.long)] = True
                if not inclusive:
                    mask_ = ~mask_
                z_.append(zh[i][mask_])
            z_ = torch.cat(z_, dim=0).unsqueeze_(0)
            return z_

        name = '_'.join([str(item) for item in items])
        z_parts = [list(set(part)) for part in parts]

        with torch.no_grad():
            items = torch.tensor(items, dtype=torch.int64, device=self.device)
            z_items, z_init, zh, gmms = self.model.get_disentanglement(items)
            z_in = merge_z()
            self.plot_sdfs(z_in, None, f'compose_{name}', verbose=True, res=256)

    def load_file(self, info_path, disclude: Optional[List[int]] = None):
        info = files_utils.load_pickle(''.join(info_path))
        keys = list(info['ids'].keys())
        items = map(lambda x: int(x.split('_')[1]) if type(x) is str else x, keys)
        items = torch.tensor(list(items), dtype=torch.int64, device=self.device)
        zh, _, gmms_sanity, _ = self.model.get_embeddings(items)
        gmms = [item for item in info['gmm']]
        zh_ = []
        split = []
        gmm_mask = torch.ones(gmms[0].shape[2], dtype=torch.bool)
        counter = 0
        # gmms_ = [[] for _ in range(len(gmms))]
        for i, key in enumerate(keys):
            gaussian_inds = info['ids'][key]
            if disclude is not None:
                for j in range(len(gaussian_inds)):
                    gmm_mask[j + counter] = gaussian_inds[j] not in disclude
                counter += len(gaussian_inds)
                gaussian_inds = [ind for ind in gaussian_inds if ind not in disclude]
                info['ids'][key] = gaussian_inds
            gaussian_inds = torch.tensor(gaussian_inds, dtype=torch.int64)
            zh_.append(zh[i, gaussian_inds])
            split.append(len(split) + torch.ones(len(info['ids'][key]), dtype=torch.int64, device=self.device))
        zh_ = torch.cat(zh_, dim=0).unsqueeze(0).to(self.device).detach().clone()  #TODO added detach,clone
        gmms = [item[:, :, gmm_mask].to(self.device) for item in info['gmm']]
        return zh_, gmms, split, info['ids']

    def load_file_doriro(self, info_path, included_list, disclude: Optional[List[int]] = None):
        info = files_utils.load_pickle(''.join(info_path))
        keys = list(info['ids'].keys())
        items = map(lambda x: int(x.split('_')[1]) if type(x) is str else x, keys)
        items = torch.tensor(list(items), dtype=torch.int64, device=self.device)
        zh, _, gmms_sanity, _ = self.model.get_embeddings(items)
        gmms = [item for item in info['gmm']]
        zh_ = []
        split = []
        gmm_mask = torch.zeros(gmms[0].shape[2], dtype=torch.bool)
        for i in included_list:
            gmm_mask[i] = 1
        counter = 0
        # gmms_ = [[] for _ in range(len(gmms))]
        for i, key in enumerate(keys):
            gaussian_inds = info['ids'][key]
            if disclude is not None:
                for j in range(len(gaussian_inds)):
                    gmm_mask[j + counter] = gaussian_inds[j] not in disclude
                counter += len(gaussian_inds)
                gaussian_inds = [ind for ind in gaussian_inds if ind not in disclude]
                info['ids'][key] = gaussian_inds
            gaussian_inds = torch.tensor(gaussian_inds, dtype=torch.int64)
            zh_.append(zh[i, gaussian_inds])
            split.append(len(split) + torch.ones(len(info['ids'][key]), dtype=torch.int64, device=self.device))
        zh_ = torch.cat(zh_, dim=0).unsqueeze(0).to(self.device)
        gmms = [item[:, :, gmm_mask].to(self.device) for item in info['gmm']]
        return zh_, gmms, split, info['ids']

    @models_utils.torch_no_grad
    def get_z_from_file(self, info_path):
        zh_, gmms, split, _ = self.load_file(info_path)
        zh_ = self.model.merge_zh_step_a(zh_, [gmms])
        zh, _ = self.model.affine_transformer.forward_with_attention(zh_)
        # gmms_ = [torch.cat(item, dim=1).unsqueeze(0) for item in gmms_]
        # zh, _ = self.model.merge_zh(zh_, [gmms])
        return zh, zh_, gmms, torch.cat(split)

    def plot_from_info(self, info_path, res):
        zh, zh_, gmms, split = self.get_z_from_file(info_path)
        mesh = self.get_mesh(zh[0], res, gmms)
        if mesh is not None:
            attention = self.get_attention_faces(mesh, zh, fixed_z=split)
        else:
            attention = None
        return mesh, attention

    def get_mesh_interpolation(self, z: T, res: int, mask: TN, alpha: T) -> Optional[T_Mesh]:

        def forward(x: T) -> T:
            nonlocal z, alpha
            x = x.unsqueeze(0)
            out = self.model.occ_head(x, z, mask=mask, alpha=alpha)[0, :]
            out = 2 * out.sigmoid_() - 1
            return out

        mesh = self.meshing.occ_meshing(forward, res=res)
        return mesh

    def get_mesh_interpolation_multiple(self, z: T, res: int, mask: TN, alpha: T) -> Optional[T_Mesh]:

        def forward(x: T) -> T:
            nonlocal z, alpha
            x = x.unsqueeze(0)
            out = self.model.occ_head(x, z, mask=mask, alpha=alpha)[0, :]
            out = 2 * out.sigmoid_() - 1
            return out

        mesh = self.meshing.occ_meshing(forward, res=res)
        return mesh

    @staticmethod
    def combine_and_pad(zh_a: T, zh_b: T) -> Tuple[T, TN]:
        if zh_a.shape[1] == zh_b.shape[1]:
            mask = None
        else:
            pad_length = max(zh_a.shape[1], zh_b.shape[1])
            mask = torch.zeros(2, pad_length, device=zh_a.device, dtype=torch.bool)
            padding = torch.zeros(1, abs(zh_a.shape[1] - zh_b.shape[1]), zh_a.shape[-1], device=zh_a.device)
            if zh_a.shape[1] > zh_b.shape[1]:
                mask[1, zh_b.shape[1]:] = True
                zh_b = torch.cat((zh_b, padding), dim=1)
            else:
                mask[0, zh_a.shape[1]:] = True
                zh_a = torch.cat((zh_a, padding), dim=1)
        return torch.cat((zh_a, zh_b), dim=0), mask

    @staticmethod
    def combine_and_pad_doriro(zh_a: T, zh_b: T, zh_c: T) -> Tuple[T, TN]:
        if zh_a.shape[1] == zh_b.shape[1] and zh_a.shape[1] == zh_c.shape[1]:
            mask = None
        else:
            pad_length = max(zh_a.shape[1], zh_b.shape[1], zh_c.shape[1])
            mask = torch.zeros(3, pad_length, device=zh_a.device, dtype=torch.bool)
            if (zh_a.shape[1] < pad_length):
                padding_a = torch.zeros(1, abs(zh_a.shape[1] - pad_length), zh_a.shape[-1], device=zh_a.device)
                mask[0, zh_a.shape[1]:] = True
                zh_a = torch.cat((zh_a, padding_a), dim=1)
            if (zh_b.shape[1] < pad_length):
                padding_b = torch.zeros(1, abs(zh_b.shape[1] - pad_length), zh_a.shape[-1], device=zh_a.device)
                mask[1, zh_a.shape[1]:] = True
                zh_b = torch.cat((zh_b, padding_b), dim=1)
            if (zh_c.shape[1] < pad_length):
                padding_c = torch.zeros(1, abs(zh_c.shape[1] - pad_length), zh_a.shape[-1], device=zh_a.device)
                mask[1, zh_a.shape[1]:] = True
                zh_c = torch.cat((zh_c, padding_c), dim=1)
            # padding = torch.zeros(1, abs(zh_a.shape[1] - zh_b.shape[1]), zh_a.shape[-1], device=zh_a.device)
            # if zh_a.shape[1] > zh_b.shape[1]:
            #     mask[1, zh_b.shape[1]:] = True
            #     zh_b = torch.cat((zh_b, padding), dim=1)
            # else:
            #     mask[0, zh_a.shape[1]: ] = True
            #     zh_a = torch.cat((zh_a, padding), dim=1)
        return torch.cat((zh_a, zh_b, zh_c), dim=0), mask

    @staticmethod
    def combine_and_pad_doriro_multiple(zh_list: List[T]) -> Tuple[T, TN]:
        zh_length = len(zh_list)
        pad_length = max(zh.shape[1] for zh in zh_list)
        if pad_length == min(zh.shape[1] for zh in zh_list):
            mask = None
        else:
            # pad_length = max(zh_a.shape[1], zh_b.shape[1], zh_c.shape[1])
            mask = torch.zeros(zh_length, pad_length, device=zh_list[0].device, dtype=torch.bool)
            for i, zh in enumerate(zh_list):
                if zh.shape[1] < pad_length:
                    padding = torch.zeros(1, abs(zh.shape[1] - pad_length), zh.shape[-1], device=zh.device)
                    mask[i, zh.shape[1]:] = True
                    zh_list[i] = torch.cat((zh, padding), dim=1)
        return torch.cat(([zh for zh in zh_list]), dim=0), mask

    @staticmethod
    def get_intersection_z(z_a: T, z_b: T) -> T:
        diff = (z_a[0, :, None, :] - z_b[0, None]).abs().sum(-1)
        diff_a = diff.min(1)[0].lt(.1)
        diff_b = diff.min(0)[0].lt(.1)
        if diff_a.shape[0] != diff_b.shape[0]:
            padding = torch.zeros(abs(diff_a.shape[0] - diff_b.shape[0]), device=z_a.device, dtype=torch.bool)
            if diff_a.shape[0] > diff_b.shape[0]:
                diff_b = torch.cat((diff_b, padding))
            else:
                diff_a = torch.cat((diff_a, padding))
        return torch.cat((diff_a, diff_b))

    @models_utils.torch_no_grad
    def get_attention_faces(self, mesh: T_Mesh, zh: T, mask: TN = None, fixed_z: TN = None, alpha: TN = None):
        coords = mesh[0][mesh[1]].mean(1).unsqueeze(0).to(zh.device)
        # coords = mesh[0].unsqueeze(0).to(zh.device)
        attention = self.model.occ_head.forward_attention(coords, zh, mask=mask, alpha=alpha)
        attention = torch.stack(attention, 0).mean(0).mean(-1)
        attention = attention.permute(1, 0, 2).reshape(attention.shape[1], -1)
        attention_max = attention.argmax(-1)
        if fixed_z is not None:
            attention_select = fixed_z[attention_max].cpu()
        else:
            attention_select = attention_max
        # colors = torch.zeros(attention_select.shape[0], 3)
        # colors[~attention_select] = torch.tensor((1., 0, 0))
        return attention_select

    def interpolate_from_files(self, item_a: Union[str, int], item_b: Union[str, int], num_mid: int, res: int,
                               counter: int = 0, logger: Optional[train_utils.Logger] = None):
        zh_a, zh_a_raw, _, _ = self.get_z_from_file(item_a)
        zh_b, zh_b_raw, _, _ = self.get_z_from_file(item_b)
        folder = files_utils.split_path(item_a)[0].split('/')[-1]
        fixed_z = self.get_intersection_z(zh_a_raw, zh_b_raw)
        zh, mask = self.combine_and_pad(zh_a, zh_b)
        if logger is None:
            logger = train_utils.Logger().start(num_mid)
        for i, alpha_ in enumerate(torch.linspace(0, 1, num_mid)):
            alpha = torch.tensor([1., 1.], device=self.device)
            alpha[0] = 1 - alpha_
            alpha[1] = alpha_
            out_path = f"{self.opt.cp_folder}/{folder}/trial_{counter + i:03d}.obj"
            if not files_utils.is_file(out_path) or True:
                mesh = self.get_mesh_interpolation(zh, res, mask, alpha)
                colors = self.get_attention_faces(mesh, zh, mask, fixed_z, alpha)
                colors = (~colors).long()
                files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/{folder}/trial_{counter + i:03d}")
                files_utils.export_list(colors.tolist(), f"{self.opt.cp_folder}/{folder}/trial_{counter + i:03d}_faces")
            logger.reset_iter()

    def interpolate_from_files_doriro(self, item_a: Union[str, int], item_b: Union[str, int], alpha_x: float, res: int,
                                      counter: int = 0, logger: Optional[train_utils.Logger] = None):
        zh_a, zh_a_raw, _, _ = self.get_z_from_file(item_a)
        zh_b, zh_b_raw, _, _ = self.get_z_from_file(item_b)
        folder = files_utils.split_path(item_a)[0].split('/')[-1]
        fixed_z = self.get_intersection_z(zh_a_raw, zh_b_raw)
        zh, mask = self.combine_and_pad(zh_a, zh_b)
        if logger is None:
            logger = train_utils.Logger().start(1)
        # for i, alpha_ in enumerate(torch.linspace(0, 1, num_mid)):
        alpha = torch.tensor([1., 1.], device=self.device)
        alpha[0] = 1 - alpha_x
        alpha[1] = alpha_x
        out_path = f"{self.opt.cp_folder}/{folder}/trial_{counter + 1:03d}.obj"
        if not files_utils.is_file(out_path) or True:
            mesh = self.get_mesh_interpolation(zh, res, mask, alpha)
            colors = self.get_attention_faces(mesh, zh, mask, fixed_z, alpha)
            colors = (~colors).long()
            files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/{folder}/trial_alpha_{alpha_x}")
            files_utils.export_list(colors.tolist(), f"{self.opt.cp_folder}/{folder}/trial_alpha_{alpha_x}_faces")
        logger.reset_iter()

    def interpolate_from_files_doriro_three(self, item_a: Union[str, int], item_b: Union[str, int],
                                            item_c: Union[str, int], alpha_a: float, alpha_b: float, alpha_c: float,
                                            res: int,
                                            counter: int = 0, logger: Optional[train_utils.Logger] = None):
        zh_a, zh_a_raw, _, _ = self.get_z_from_file(item_a)
        zh_b, zh_b_raw, _, _ = self.get_z_from_file(item_b)
        zh_c, zh_c_raw, _, _ = self.get_z_from_file(item_c)
        folder = files_utils.split_path(item_a)[0].split('/')[-1]
        # fixed_z = self.get_intersection_z_doriro(zh_a_raw, zh_b_raw, zh_c_raw)
        zh, mask = self.combine_and_pad_doriro(zh_a, zh_b, zh_c)
        if logger is None:
            logger = train_utils.Logger().start(1)
        # for i, alpha_ in enumerate(torch.linspace(0, 1, num_mid)):
        alpha = torch.tensor([1., 1., 1.], device=self.device)
        alpha[0] = alpha_a
        alpha[1] = alpha_b
        alpha[2] = alpha_c
        out_path = f"{self.opt.cp_folder}/{folder}/trial_{counter + 1:03d}.obj"
        if not files_utils.is_file(out_path) or True:
            mesh = self.get_mesh_interpolation_multiple(zh, res, mask, alpha)
            # colors = self.get_attention_faces(mesh, zh, mask, fixed_z, alpha)
            # colors = (~colors).long()
            files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/{folder}/trial_alpha_multiple")
            # files_utils.export_list(colors.tolist(), f"{self.opt.cp_folder}/{folder}/trial_alpha_{alpha_x}_faces")
        logger.reset_iter()

    def interpolate_from_files_doriro_multiple(self, items: List[Union[str, int]], alphas: List[float], res: int,
                                               logger: Optional[train_utils.Logger] = None):
        folder = files_utils.split_path(items[0])[0].split('/')[-1]
        zh, mask = self.combine_and_pad_doriro_multiple([self.get_z_from_file(item)[0] for item in items])
        if logger is None:
            logger = train_utils.Logger().start(1)
        # for i, alpha_ in enumerate(torch.linspace(0, 1, num_mid)):
        alpha = torch.tensor([1. for item in items], device=self.device)
        for i in range(len(alphas)):
            alpha[i] = alphas[i]
        mesh = self.get_mesh_interpolation_multiple(zh, res, mask, alpha)
        files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/{folder}/trial_alpha_multiple")
        logger.reset_iter()

    def interpolate_from_files_doriro_multiple_alphas(self, items: List[Union[str, int]], alphas: List[List[float]],
                                                      res: int,
                                                      logger: Optional[train_utils.Logger] = None):
        folder = files_utils.split_path(items[0])[0].split('/')[-1]
        zh, mask = self.combine_and_pad_doriro_multiple([self.get_z_from_file(item)[0] for item in items])
        if logger is None:
            logger = train_utils.Logger().start(1)
        # for i, alpha_ in enumerate(torch.linspace(0, 1, num_mid)):
        alpha = torch.tensor([1. for item in items], device=self.device)
        for i in range(len(alphas)):
            alpha[i] = alphas[i]
        mesh = self.get_mesh_interpolation_multiple(zh, res, mask, alpha)
        files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/{folder}/trial_alpha_multiple")
        logger.reset_iter()

    @models_utils.torch_no_grad
    def plot_folder(self, *folders, res: int = 256):
        logger = train_utils.Logger()
        for folder in folders:
            paths = files_utils.collect(folder, '.pkl')
            logger.start(len(paths))
            for path in paths:
                name = path[1]
                out_path = f"{self.opt.cp_folder}/from_ui/{name}"
                mesh, colors = self.plot_from_info(path, res)
                if mesh is not None:
                    files_utils.export_mesh(mesh, out_path)
                    files_utils.export_list(colors.tolist(), f"{out_path}_faces")
                logger.reset_iter()
            logger.stop()

    def get_samples_by_names(self, names: List[str]) -> T:
        ds = mesh_datasets.CacheDataset(self.opt.dataset_name, self.opt.num_samples, self.opt.data_symmetric)
        return torch.tensor([ds.get_item_by_name(name) for name in names], dtype=torch.int64)

    def get_names_by_samples(self, items):
        ds = mesh_datasets.CacheDataset(self.opt.dataset_name, self.opt.num_samples, self.opt.data_symmetric)
        res = [ds.get_name(item) for item in items]
        return [ds.get_name(item) for item in items]

    def get_zh_from_idx(self, items: T):
        zh, _, gmms, __ = self.model.get_embeddings(items.to(self.device))
        zh, attn_b = self.model.merge_zh(zh, gmms)
        return zh, gmms

    @models_utils.torch_no_grad
    def plot(self, prefix: Union[str, int], verbose=False, interpolate: bool = False, res: int = 200, size: int = -1,
             names: Optional[List[str]] = None):
        if size <= 0:
            size = self.opt.batch_size // 2
        if names is not None:
            fixed_items = self.get_samples_by_names(names)
        elif self.model.opt.dataset_size < size:
            fixed_items = torch.arange(self.model.opt.dataset_size)
        else:
            if self.model.stash is None:
                numbers = files_utils.collect(f"{constants.CHECKPOINTS_ROOT}/occ_gmm_chairs_sym_hard/occ/", '.obj')
                numbers = list(map(lambda x: int(x[1].split('_')[1]), numbers))
                fixed_items = torch.randint(low=0, high=self.opt.dataset_size, size=(size,))
                # fixed_items = torch.tensor(numbers)
                # fixed_items = torch.randint(low=0, high=self.model.stash.shape[0], size=(size,))
            else:
                fixed_items = torch.randint(low=0, high=self.model.stash.shape[0], size=(size,))
                # fixed_items = torch.randint(low=0, high=self.model.stash.shape[0], size=(size,))
        # if names is None and self.model.stash is None:
        # names = self.get_names_by_samples(fixed_items)
        # else:
        # names = ["random"] * len(fixed_items)
        print()
        # numbers = [i for i in range(2)]
        numbers = [755, 2646]
        fixed_items = torch.tensor(numbers)  # TODO DORI
        names = [f'{i:02d}_' for i in fixed_items]
        # fixed_items = torch.arange(size) + 100
        # fixed_items = fixed_items.unique()
        # fixed_items = torch.tensor(fixed_items).long().to(self.device)
        if type(prefix) is int:
            prefix = f'{prefix:04d}'
        if interpolate:
            item_a = torch.tensor(numbers[0])
            item_b = torch.tensor(numbers[1])

            zh, gmms = self.model.interpolate(item_a, item_b, num_between=3)
            use_item_id = False
        else:
            zh, _, gmms, attn_a = self.model.get_embeddings(fixed_items.to(self.device))
            zh, attn_b = self.model.merge_zh(zh, gmms)
            use_item_id = True
        self.plot_occ(zh, gmms, prefix, verbose=verbose, use_item_id=use_item_id,
                      res=res, fixed_items=names)

    # 1617, 3148, 2175, 529, 1435, 660, 553, 719, 679

    @models_utils.torch_no_grad
    def plot_doriro(self, chair_ids: List[int], prefix: Union[str, int] = "sample", verbose=False,
                    interpolate: bool = False, res: int = 200, size: int = -1,
                    names: Optional[List[str]] = None):
        numbers = chair_ids
        fixed_items = torch.tensor(numbers)
        names = [f'{i:02d}_' for i in fixed_items]
        if type(prefix) is int:
            prefix = f'{prefix:04d}'
        zh, _, gmms, attn_a = self.model.get_embeddings(fixed_items.to(self.device))
        zh, attn_b = self.model.merge_zh(zh, gmms)
        use_item_id = True
        # self.save_light(opt_.cp_folder, gmms)
        self.plot_occ(zh, gmms, prefix, verbose=verbose, use_item_id=use_item_id,
                      res=res, fixed_items=names)

    @models_utils.torch_no_grad
    def plot_doriro_flask(self, chair_ids: List[int], prefix: Union[str, int] = "sample", verbose=False,
                          interpolate: bool = False, res: int = 200, size: int = -1,
                          names: Optional[List[str]] = None):
        numbers = chair_ids
        fixed_items = torch.tensor(numbers)
        names = [f'{i:02d}_' for i in fixed_items]
        if type(prefix) is int:
            prefix = f'{prefix:04d}'
        zh, _, gmms, attn_a = self.model.get_embeddings(fixed_items.to(self.device))
        zh, attn_b = self.model.merge_zh(zh, gmms)
        use_item_id = True
        # self.save_light(opt_.cp_folder, gmms)
        return self.plot_occ(zh, gmms, prefix, verbose=verbose, use_item_id=use_item_id,
                             res=res, fixed_items=names)

    @models_utils.torch_no_grad
    def plot_random_doriro(self, amount: int = 1, prefix: Union[str, int] = "sample", verbose=False, res: int = 200,
                           names: Optional[List[str]] = None, noise: float = 0):
        z = self.model.get_random_embeddings(amount, noise, only_noise=True)
        zh, gmms, attn = self.model.occ_former(z)

        # mesh = self.get_mesh(zh_[0], res, None)
        # files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/occ_mix/mix_{i:02d}")

        # numbers = chair_ids
        # fixed_items = torch.tensor(numbers)
        # names = [f'{i:02d}_' for i in fixed_items]
        # if type(prefix) is int:
        #     prefix = f'{prefix:04d}'
        # zh, _, gmms, attn_a = self.model.get_embeddings(fixed_items.to(self.device))
        zh, attn_b = self.model.merge_zh(zh, gmms)
        use_item_id = False
        # self.save_light(opt_.cp_folder, gmms)
        self.plot_occ(zh, gmms, prefix, verbose=verbose, use_item_id=use_item_id,
                      res=res, fixed_items=names)

    def plot_mix(self):
        # fixed_items = torch.randint(low=0, high=self.opt.dataset_size, size=(self.opt.batch_size,))
        # numbers = [755, 2646, 3109]
        numbers = [755, 2646]
        fixed_items = torch.tensor(numbers)  # TODO DORI
        with torch.no_grad():
            z, _, gmms, extra = self.model.get_embeddings(fixed_items.to(self.device))
            z = self.mix_z(gmms, z)
            self.plot_occ(z, None, "mix", verbose=True, use_item_id=True, res=100, fixed_items=fixed_items)

    def interpolate_seq(self, num_mid: int, *seq: str, res=200):
        logger = train_utils.Logger().start((len(seq) - 1) * num_mid)
        for i in range(len(seq) - 1):
            self.interpolate_from_files(seq[i], seq[i + 1], num_mid, res, i * num_mid, logger=logger)
        logger.stop()

    def interpolate(self, item_a: Union[str, int], item_b: Union[str, int], num_mid: int, res: int = 200):
        if type(item_a) is str:
            self.interpolate_from_files(item_a, item_b, num_mid, res)
        else:
            zh = self.model.interpolate(item_a, item_b, num_mid)[0]
            for i in range(num_mid):
                mesh = self.get_mesh(zh[i], res)
                files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/interpolate/{item_a}_{item_b}_{i}")
                print(f'done {i + 1:d}/{num_mid}')

    def interpolate_doriro(self, item_a: Union[str, int], item_b: Union[str, int], num_mid: int, res: int = 200):
        if type(item_a) is str:
            self.interpolate_from_files(item_a, item_b, num_mid, res)
        else:
            # zh = self.model.interpolate(item_a, item_b, num_mid)[0]
            zh, gmm = self.model.interpolate(item_a, item_b, num_mid)
            for i in range(num_mid):
                mesh = self.get_mesh(zh[i], res, gmm)
                files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/interpolate/{item_a}_{item_b}_{i}")
                print(f'done {i + 1:d}/{num_mid}')

    @property
    def device(self):
        return self.opt.device

    def measure_time(self, num_samples: int, *res: int):

        fixed_items = torch.randint(low=0, high=self.opt.dataset_size, size=(num_samples,))
        zh, _, gmms, attn_a = self.model.get_embeddings(fixed_items.to(self.device))
        zh, attn_b = self.model.merge_zh(zh, gmms)
        for res_ in res:
            print(f"\nmeasure {res_:d}")
            times_a, times_b = [], []
            for i in range(len(zh)):
                time_a, time_b = self.get_mesh(zh[i], res_, get_time=True)
                if i > 1:
                    times_a.append(time_a)
                    times_b.append(time_b)
            times_a = torch.tensor(times_a).float()
            times_b = torch.tensor(times_b).float()
            for times in (times_a, times_b):
                print(f"avg: {times.mean()}, std: {times.std()}, min: {times.min()}, , max: {times.max()}")

    @models_utils.torch_no_grad
    def random_stash(self, nums_sample: int, name: str):
        z = self.model.get_random_embeddings(nums_sample).detach().cpu()
        files_utils.save_pickle(z, f'{self.opt.cp_folder}/{name}')

    def load_random(self, name: str):
        z = files_utils.load_pickle(f'{self.opt.cp_folder}/{name}')
        self.model.stash_embedding(z)

    @models_utils.torch_no_grad
    def random_samples(self, nums_sample, res=256):
        logger = train_utils.Logger().start(nums_sample)
        num_batches = nums_sample // self.opt.batch_size + int((nums_sample % self.opt.batch_size) != 0)
        counter = 0
        for batch in range(num_batches):
            if batch == num_batches - 1:
                batch_size = nums_sample - counter
            else:
                batch_size = self.opt.batch_size
            zh, gmms = self.model.random_samples(batch_size)
            for i in range(len(zh)):
                # gmm_ = [gmms[j][i].unsqueeze(0) for j in range(len(gmms))]
                mesh = self.get_mesh(zh[i], res, None)
                pcd = mesh_utils.sample_on_mesh(mesh, 2048, sample_s=mesh_utils.SampleBy.AREAS)[0]
                files_utils.save_np(pcd, f'{self.opt.cp_folder}/gen/pcd_{counter:04d}')
                files_utils.export_mesh(mesh, f'{self.opt.cp_folder}/gen/{counter:04d}')
                logger.reset_iter()
                counter += 1
        logger.stop()
        # self.plot_occ(zh, gmms, prefix, res=res, verbose=True)

    def get_plot_scale(self):
        scale = files_utils.load_pickle(f'{constants.CACHE_ROOT}{self.opt.dataset_name}/scale')
        if scale is None:
            return 1.
        return scale['global_max'] / scale['std']

    def plot_from_file(self, item_a: int, item_b: int):
        data = files_utils.load_pickle(f"{self.opt.cp_folder}/compositions/{item_a:d}_{item_b:d}")
        (item_a, gmms_id_a), (item_b, gmms_id_b) = data
        self.disentanglement_plot(item_a, item_b, gmms_id_a, gmms_id_b, b_inclusive=True)

    def plot_from_file_single(self, item_a: int, item_b: int, in_item: int):
        data = files_utils.load_pickle(f"{self.opt.cp_folder}/compositions/{item_a:d}_{item_b:d}")
        (item_a, gmms_id_a) = data[in_item]
        self.disentanglement_plot(item_a, -1, gmms_id_a, [], b_inclusive=True)

    def get_mesh_from_mid(self, gmm, included: T, res: int) -> Optional[T_Mesh]:
        if self.mid is None:
            return None
        gmm = [elem.to(self.device) for elem in gmm]
        included = included.to(device=self.device)
        print(included[:, 0], included[:, 1])
        mid_ = self.mid[included[:, 0], included[:, 1]].unsqueeze(0)
        zh = self.model.merge_zh(mid_, gmm)[0]
        # zh = self.model_a.merge_zh(self.mid, gmm, mask=mask)[0]
        # zh = zh[included[0]]
        mesh = self.get_mesh(zh[0], res, [gmm])
        return mesh

    def set_items(self, *items: int):
        items = torch.tensor(items, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            self.mid = self.model.forward_a(items)[0]

    def set_items_get_all(self, *items: int):
        items = torch.tensor(items, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            return self.model.forward_a(items)

    def save_light(self, root, gmms):
        gmms = self.sort_gmms(*gmms)
        save_dict = {'ids': {
            gmm.shape_id: [gaussian.gaussian_id[1] for gaussian in gmm.gmm if gaussian.included]
            # for gmm in self.gmms if gmm.included},
            for gmm in gmms},  # TODO doriro
            'gmm': gmms}
        path = f"{root}/{files_utils.get_time_name('light')}"
        files_utils.save_pickle(save_dict, path)

    def sort_gmms(self, gmms):
        order = torch.arange(gmms[0].shape[2]).tolist()
        # order = sorted(order, key=lambda x: included[x][0] * 100 + included[x][1])
        gmms = [[item[:, :, order[i]] for item in gmms] for i in range(gmms[0].shape[2])]
        gmms = [torch.stack([gmms[j][i] for j in range(len(gmms))], dim=2) for i in range(len(gmms[0]))]
        return gmms

    @models_utils.torch_no_grad
    def mix_file(self, path: str, z_idx, replace_inds, num_samples: int, res=220):
        '''
        Randomize a part of the file (inserted as a pkl).
        :param path:
        :param z_idx:
        :param replace_inds:
        :param num_samples:
        :param res:
        :return:
        '''
        zh, gmms, _, base_inds = self.load_file(path, disclude=z_idx)
        # select = torch.ones(zh.shape[1], dtype=torch.bool)
        # select[torch.tensor(z_idx, dtype=torch.int64)] = False
        replace_inds_t = torch.tensor(replace_inds, dtype=torch.int64, device=self.device)
        # select = select.to(self.device)

        # gmms = [item[0, :, select] for item in gmms]
        # select_random = torch.rand(1000).argsort()[:num_samples]
        select_random = torch.rand(1000)
        select_random = select_random.argsort()
        select_random = select_random[:num_samples]
        # select_random = (1.1 - 0.9) * select_random + 0.9
        zh_new, _, gmms_new, _ = self.model.get_embeddings(select_random.to(self.device))
        zh_new = zh_new[:, replace_inds_t]
        gmms_new = [item[:, :, replace_inds_t] for item in gmms_new[0]]
        name = files_utils.get_time_name('light')
        for i in range(num_samples):
            zh_ = torch.cat((zh, zh_new[i].unsqueeze(0)), dim=1)
            gmms_ = [torch.cat((item, item_new[i].unsqueeze(0)), dim=2) for item, item_new in zip(gmms, gmms_new)]
            zh_ = self.model.merge_zh_step_a(zh_, [gmms_])
            zh_, _ = self.model.affine_transformer.forward_with_attention(zh_)
            mesh = self.get_mesh(zh_[0], res, None)
            files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/occ_mix/mix_{i:02d}")
            # save_dict = {'ids': base_inds | {select_random[i].item(): replace_inds},
            #              'gmm': [item.cpu() for item in gmms_]}
            # path = f"{self.opt.cp_folder}/occ_mix_light/{name}_{i:02d}"
            # files_utils.save_pickle(save_dict, path)
            # files_utils.export_gmm()

    @models_utils.torch_no_grad
    def mix_2_files_doriro(self, path_1: str, path_2: str, path_3: str, replace_inds, z_idx, num_samples: int, res=220):
        '''
        Replace indices of 1 pkl with another.
        :param path:
        :param z_idx:
        :param replace_inds:
        :param num_samples:
        :param res:
        :return:
        '''
        zh, gmms, _, base_inds = self.load_file(path_1, disclude=z_idx)
        print(path_1)
        print(path_2)
        print(path_3)
        zh_2, gmms_2, _2, base_inds_2 = self.load_file(path_2)
        zh_3, gmms_3, _3, base_inds_3 = self.load_file(path_3)
        # select = torch.ones(zh.shape[1], dtype=torch.bool)
        # select[torch.tensor(z_idx, dtype=torch.int64)] = False
        replace_inds_t = torch.tensor(replace_inds, dtype=torch.int64, device=self.device)
        print("replacing indices: ", replace_inds)
        # select = select.to(self.device)

        zh_2 = zh_2[:, replace_inds_t]
        zh_3 = zh_3[:, replace_inds_t]
        #
        # zh_2 = 0.5*zh_2 + 0.5*zh_3
        # for i in range(len(gmms_2)):
        #     gmms_2[i] = 0.5*gmms_2[i] + 0.5*gmms_3[i]
        # gmms_new = [item[:, :, replace_inds_t] for item in gmms_new[0]]
        gmms_new = [item[:, :, replace_inds_t] for item in gmms_2]
        name = files_utils.get_time_name('light')

        zh_ = torch.cat((zh, zh_2[0].unsqueeze(0)), dim=1)
        gmms_ = [torch.cat((item, item_new[0].unsqueeze(0)), dim=2) for item, item_new in zip(gmms, gmms_new)]
        zh_ = self.model.merge_zh_step_a(zh_, [gmms_])
        zh_, _ = self.model.affine_transformer.forward_with_attention(zh_)
        mesh = self.get_mesh(zh_[0], res, None)
        files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/occ_mix/mix")
        # save_dict = {'ids': base_inds | {select_random[i].item(): replace_inds},
        #              'gmm': [item.cpu() for item in gmms_]}
        # path = f"{self.opt.cp_folder}/occ_mix_light/{name}_{i:02d}"
        # files_utils.save_pickle(save_dict, path)
        # files_utils.export_gmm()

    @models_utils.torch_no_grad
    def mix_multiple_files_doriro(self, paths: List[str], alphas: List[List[float]], res=220):
        '''
        Mix files together.
        '''

        # create an empty gmm and start adding to it.
        empty_zh, empty_gmms, empty_, empty_base_inds = self.load_file(paths[0], disclude=[i for i in range(16)])

        zh_list, gmms_list, _list, base_inds_list = ([] for i in range(4))
        for path in paths:
            zh, gmms, _, base_inds = self.load_file(path)
            zh_list.append(zh)
            gmms_list.append(gmms)
            _list.append(_)
            base_inds_list.append(base_inds)

        # mix all given files according to alphas

        min_gmm_length = min(gmm[0].shape[2] for gmm in gmms_list)

        combined_added_gmm = gmms_list[0]
        # for i in range(len(combined_added_gmm)):
        for i in range(16):
            # maybe some padding is needed to keep all gmms at 16.
            for k in range(4):
                if k == 0 or k == 3:
                    combined_added_gmm[k][:, :, i, :] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i, :] for j in range(len(gmms_list)))
                elif k == 1:
                    combined_added_gmm[k][:, :, i, :, :] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i, :, :] for j in range(len(gmms_list)))
                elif k == 2:
                    combined_added_gmm[k][:, :, i] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i] for j in range(len(gmms_list)))

        # gmms_new = [item[:, :, replace_inds_t] for item in gmms_new[0]]
        # gmms_new = [item[:, :, replace_inds_t] for item in gmms_2]
        # name = files_utils.get_time_name('light')

        gmms_new = combined_added_gmm
        # make zh something that makes sense.

        # option 1 - multiply each zh component with correct alpha and then combine them all.
        unsqueezed_zh = [zh[0].unsqueeze(0) for zh in zh_list]
        for i in range(16):
            unsqueezed_zh[0][:, i, :] = sum(alphas[j][i] * unsqueezed_zh[j][:, i, :] for j in range(len(unsqueezed_zh)))

        # option 2 - just make it 0.5 for all alphas.
        # option 3 - previous.
        # zh_ = torch.cat((empty_zh, *unsqueezed_zh), dim=1)
        zh_ = unsqueezed_zh[0]
        gmms_ = [torch.cat((item, item_new[0].unsqueeze(0)), dim=2) for item, item_new in zip(empty_gmms, gmms_new)]
        zh_ = self.model.merge_zh_step_a(zh_, [gmms_])
        zh_, _ = self.model.affine_transformer.forward_with_attention(zh_)
        mesh = self.get_mesh(zh_[0], res, None)
        files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/occ_mix/mix")

    @models_utils.torch_no_grad
    def mix_multiple_files_doriro_flask(self, indices: List[int], alphas: List[List[float]], res=220):
        '''
        Mix files together.
        '''

        # create an empty gmm and start adding to it.
        # empty_zh, empty_gmms, empty_, empty_base_inds = self.load_file(paths[0], disclude=[i for i in range(16)])
        print("length is", len(self.loaded_files_dict))
        empty_zh, empty_gmms, empty_, empty_base_inds = self.loaded_files_dict.get(0)

        zh_list, gmms_list, _list, base_inds_list = ([] for i in range(4))
        for i in indices:
            zh, gmms, _, base_inds = self.loaded_files_dict.get(i)
            zh_list.append(zh)
            gmms_list.append(gmms)
            _list.append(_)
            base_inds_list.append(base_inds)

        # mix all given files according to alphas

        combined_added_gmm = gmms_list[0]
        # for i in range(len(combined_added_gmm)):
        for i in range(16):
            # maybe some padding is needed to keep all gmms at 16.
            for k in range(4):
                if k == 0 or k == 3:
                    combined_added_gmm[k][:, :, i, :] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i, :] for j in range(len(gmms_list)))
                elif k == 1:
                    combined_added_gmm[k][:, :, i, :, :] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i, :, :] for j in range(len(gmms_list)))
                elif k == 2:
                    combined_added_gmm[k][:, :, i] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i] for j in range(len(gmms_list)))

        # gmms_new = [item[:, :, replace_inds_t] for item in gmms_new[0]]
        # gmms_new = [item[:, :, replace_inds_t] for item in gmms_2]
        # name = files_utils.get_time_name('light')

        gmms_new = combined_added_gmm
        # make zh something that makes sense.

        # option 1 - multiply each zh component with correct alpha and then combine them all.
        unsqueezed_zh = [zh[0].unsqueeze(0) for zh in zh_list]
        for i in range(16):
            unsqueezed_zh[0][:, i, :] = sum(alphas[j][i] * unsqueezed_zh[j][:, i, :] for j in range(len(unsqueezed_zh)))

        # option 2 - just make it 0.5 for all alphas.
        # option 3 - previous.
        # zh_ = torch.cat((empty_zh, *unsqueezed_zh), dim=1)
        zh_ = unsqueezed_zh[0]
        gmms_ = [torch.cat((item, item_new[0].unsqueeze(0)), dim=2) for item, item_new in zip(empty_gmms, gmms_new)]
        zh_ = self.model.merge_zh_step_a(zh_, [gmms_])
        zh_, _ = self.model.affine_transformer.forward_with_attention(zh_)
        mesh = self.get_mesh(zh_[0], res, None)
        # return files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/occ_mix/mix")
        return files_utils.export_mesh(mesh, f"{self.opt.flask_public_folder}/mix")
        # return files_utils.return_obj_mesh(mesh, f"{self.opt.flask_public_folder}/mix")

    @models_utils.torch_no_grad
    def mix_multiple_files_doriro_flask_gltf(self, indices: List[int], alphas: List[List[float]], res=220):
        '''
        Mix files together.
        '''

        empty_zh, empty_gmms, empty_, empty_base_inds = self.loaded_files_dict.get(0)

        zh_list, gmms_list, _list, base_inds_list = ([] for i in range(4))
        for i in indices:
            zh, gmms, _, base_inds = self.loaded_files_dict.get(i)
            zh_list.append(zh)
            gmms_list.append(gmms)
            _list.append(_)
            base_inds_list.append(base_inds)
        # mix all given files according to alphas
        # mix all given files according to alphas

        combined_added_gmm = gmms_list[0] #TODO updated
        a = torch.tensor(np.zeros(shape=gmms_list[0][0].shape), device=self.device, dtype=torch.float32)
        b = torch.tensor(np.zeros(shape=gmms_list[0][1].shape), device=self.device, dtype=torch.float32)
        c = torch.tensor(np.zeros(shape=gmms_list[0][2].shape), device=self.device, dtype=torch.float32)
        d = torch.tensor(np.zeros(shape=gmms_list[0][3].shape), device=self.device, dtype=torch.float32)
        combined_added_gmm = [a,b,c,d]
        # combined_added_gmm = gmms_list[1]
        # for i in range(len(combined_added_gmm)):
        for i in range(16):
            # maybe some padding is needed to keep all gmms at 16.
            for k in range(4):
                if k == 0 or k == 3:
                    combined_added_gmm[k][:, :, i, :] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i, :] for j in range(len(gmms_list)))
                elif k == 1:
                    combined_added_gmm[k][:, :, i, :, :] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i, :, :] for j in range(len(gmms_list)))
                elif k == 2:
                    combined_added_gmm[k][:, :, i] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i] for j in range(len(gmms_list)))

        # gmms_new = [item[:, :, replace_inds_t] for item in gmms_new[0]]
        # gmms_new = [item[:, :, replace_inds_t] for item in gmms_2]
        # name = files_utils.get_time_name('light')

        gmms_new = combined_added_gmm
        # make zh something that makes sense.

        # option 1 - multiply each zh component with correct alpha and then combine them all.
        unsqueezed_zh = [zh_a[0].unsqueeze(0) for zh_a in zh_list]

        zh_alternative = torch.tensor(np.zeros(shape=unsqueezed_zh[0].shape), device=self.device, dtype=torch.float32)

        for i in range(16):
            zh_alternative[:, i, :] = sum(
                alphas[j][i] * unsqueezed_zh[j][:, i, :] for j in range(len(unsqueezed_zh)))

        # option 2 - just make it 0.5 for all alphas.
        # option 3 - previous.
        zh_ = zh_alternative
        # TODO added detach.clone
        gmms_ = [torch.cat((item, item_new[0].unsqueeze(0)), dim=2) for item, item_new in zip(empty_gmms, gmms_new)]
        zh_ = self.model.merge_zh_step_a(zh_, [gmms_])
        zh_, _ = self.model.affine_transformer.forward_with_attention(zh_)
        mesh = self.get_mesh(zh_[0], res, None)
        vs, faces = files_utils.return_obj_mesh(mesh, f"{self.opt.cp_folder}/mix")

        # return obj
        mesh = trimesh.Trimesh(vertices=vs, faces=faces)
        print("trying to convert obj to gltf using glb")
        result2 = trimesh.exchange.gltf.export_glb(mesh, include_normals=True)
        return result2


    @models_utils.torch_no_grad
    def mix_multiple_files_doriro_flask_archive(self, indices: List[int], alphas: List[List[float]], res=220):
        '''
        Mix files together.
        '''

        empty_zh, empty_gmms, empty_, empty_base_inds = self.loaded_files_dict.get(0)

        zh_list, gmms_list, _list, base_inds_list = ([] for i in range(4))
        for i in indices:
            zh, gmms, _, base_inds = self.loaded_files_dict.get(i)
            zh_list.append(zh)
            gmms_list.append(gmms)
            _list.append(_)
            base_inds_list.append(base_inds)
        # mix all given files according to alphas
        # mix all given files according to alphas

        combined_added_gmm = gmms_list[0] #TODO updated
        a = torch.tensor(np.zeros(shape=gmms_list[0][0].shape), device=self.device, dtype=torch.float32)
        b = torch.tensor(np.zeros(shape=gmms_list[0][1].shape), device=self.device, dtype=torch.float32)
        c = torch.tensor(np.zeros(shape=gmms_list[0][2].shape), device=self.device, dtype=torch.float32)
        d = torch.tensor(np.zeros(shape=gmms_list[0][3].shape), device=self.device, dtype=torch.float32)
        combined_added_gmm = [a,b,c,d]
        # combined_added_gmm = gmms_list[1]
        # for i in range(len(combined_added_gmm)):
        for i in range(16):
            # maybe some padding is needed to keep all gmms at 16.
            for k in range(4):
                if k == 0 or k == 3:
                    combined_added_gmm[k][:, :, i, :] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i, :] for j in range(len(gmms_list)))
                elif k == 1:
                    combined_added_gmm[k][:, :, i, :, :] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i, :, :] for j in range(len(gmms_list)))
                elif k == 2:
                    combined_added_gmm[k][:, :, i] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i] for j in range(len(gmms_list)))

        # gmms_new = [item[:, :, replace_inds_t] for item in gmms_new[0]]
        # gmms_new = [item[:, :, replace_inds_t] for item in gmms_2]
        # name = files_utils.get_time_name('light')

        gmms_new = combined_added_gmm
        # make zh something that makes sense.

        # option 1 - multiply each zh component with correct alpha and then combine them all.
        unsqueezed_zh = [zh_a[0].unsqueeze(0) for zh_a in zh_list]
        for i in range(16):
            unsqueezed_zh[0][:, i, :] = sum(
                alphas[j][i] * unsqueezed_zh[j][:, i, :] for j in range(len(unsqueezed_zh)))

        # option 2 - just make it 0.5 for all alphas.
        # option 3 - previous.
        zh_ = unsqueezed_zh[0]
        # TODO added detach.clone
        gmms_ = [torch.cat((item, item_new[0].unsqueeze(0)), dim=2) for item, item_new in zip(empty_gmms, gmms_new)]
        zh_ = self.model.merge_zh_step_a(zh_, [gmms_])
        zh_, _ = self.model.affine_transformer.forward_with_attention(zh_)
        mesh = self.get_mesh(zh_[0], res, None)
        vs, faces = files_utils.return_obj_mesh(mesh, f"{self.opt.cp_folder}/mix")

        # return obj
        mesh = trimesh.Trimesh(vertices=vs, faces=faces)
        return mesh

    @models_utils.torch_no_grad
    def mix_multiple_files_doriro_flask_json(self, indices: List[int], alphas: List[List[float]], res=220):
        '''
        Mix files together.
        '''

        # create an empty gmm and start adding to it.
        # empty_zh, empty_gmms, empty_, empty_base_inds = self.load_file(paths[0], disclude=[i for i in range(16)])
        print("length is", len(self.loaded_files_dict.get(0)))
        empty_zh, empty_gmms, empty_, empty_base_inds = self.loaded_files_dict.get(0)

        zh_list, gmms_list, _list, base_inds_list = ([] for i in range(4))
        for i in indices:
            zh, gmms, _, base_inds = self.loaded_files_dict.get(i)
            zh_list.append(zh)
            gmms_list.append(gmms)
            _list.append(_)
            base_inds_list.append(base_inds)

        # mix all given files according to alphas

        combined_added_gmm = gmms_list[0]
        # for i in range(len(combined_added_gmm)):
        for i in range(16):
            # maybe some padding is needed to keep all gmms at 16.
            for k in range(4):
                if k == 0 or k == 3:
                    combined_added_gmm[k][:, :, i, :] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i, :] for j in range(len(gmms_list)))
                elif k == 1:
                    combined_added_gmm[k][:, :, i, :, :] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i, :, :] for j in range(len(gmms_list)))
                elif k == 2:
                    combined_added_gmm[k][:, :, i] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i] for j in range(len(gmms_list)))

        # gmms_new = [item[:, :, replace_inds_t] for item in gmms_new[0]]
        # gmms_new = [item[:, :, replace_inds_t] for item in gmms_2]
        # name = files_utils.get_time_name('light')

        gmms_new = combined_added_gmm
        # make zh something that makes sense.

        # option 1 - multiply each zh component with correct alpha and then combine them all.
        unsqueezed_zh = [zh[0].unsqueeze(0) for zh in zh_list]
        for i in range(16):
            unsqueezed_zh[0][:, i, :] = sum(alphas[j][i] * unsqueezed_zh[j][:, i, :] for j in range(len(unsqueezed_zh)))

        # option 2 - just make it 0.5 for all alphas.
        # option 3 - previous.
        # zh_ = torch.cat((empty_zh, *unsqueezed_zh), dim=1)
        zh_ = unsqueezed_zh[0]
        gmms_ = [torch.cat((item, item_new[0].unsqueeze(0)), dim=2) for item, item_new in zip(empty_gmms, gmms_new)]
        zh_ = self.model.merge_zh_step_a(zh_, [gmms_])
        zh_, _ = self.model.affine_transformer.forward_with_attention(zh_)
        mesh = self.get_mesh(zh_[0], res, None)
        # return files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/occ_mix/mix")
        # return files_utils.export_mesh(mesh, f"{self.opt.flask_public_folder}/mix")
        return files_utils.return_obj_mesh(mesh, f"{self.opt.flask_public_folder}/mix")

    def plot_only_parts(self, item, item_id, included_list):
        self.set_items((item_id))
        included_list_for_tensor = [[0, i] for i in included_list]
        included = torch.tensor(included_list_for_tensor, dtype=torch.int64)
        zh_, gmms_1, split, info = inference.load_file_doriro(item, included_list)

        # Extract a gmm, create a fake "included", and see if a mesh is available. Afterwards understand where to insert the alpha
        # variable to make combinations.
        mesh = inference.get_mesh_from_mid(gmms_1, included=included, res=200)
        files_utils.export_mesh(mesh, f'{inference.opt.cp_folder}/occ_bla/bla')

    def random_gt(self):
        ds_train = mesh_datasets.CacheDataset(self.opt.dataset_name, self.opt.num_samples, self.opt.data_symmetric)
        ds_test = mesh_datasets.CacheDataset(self.opt.dataset_name.replace('train', 'test'), self.opt.num_samples,
                                             self.opt.data_symmetric)
        num_test = min(500, len(ds_test))
        num_train = 1000 - num_test
        cls = self.opt.dataset_name.split('_')[1]
        counter = 0
        for num_items, ds in zip((num_test, num_train), (ds_test, ds_train)):
            select = np.random.choice(len(ds), num_items, replace=False)
            for item in select:
                mesh_name = f'{constants.Shapenet_WT}/{cls}/{ds.get_name(item)}'
                mesh = files_utils.load_mesh(mesh_name)
                mesh = mesh_utils.to_unit_sphere(mesh, scale=.9)
                pcd = mesh_utils.sample_on_mesh(mesh, 2048, sample_s=mesh_utils.SampleBy.AREAS)[0]
                files_utils.save_np(pcd, f'{constants.CACHE_ROOT}/evaluation/generation/{cls}/gt/pcd_{counter:04d}')
                counter += 1

    def plot_single(self, *names: str, res: int = 200):
        gmms = []
        items = []
        paths = []
        included = []
        for name in names:
            paths.append(f'{self.opt.cp_folder}/single_edit/{name}')
            phi, mu, eigen, p, include = files_utils.load_gmm(paths[-1], device=self.device)
            gmms.append([item.unsqueeze(0) for item in (mu, p, phi, eigen)])
            items.append(torch.tensor([int(name.split('_')[0])], device=self.device, dtype=torch.int64))
            included.append(include)
        zh, _, _, attn_a = self.model.forward_a(torch.cat(items, dim=0))
        gmms = [torch.stack(elem) for elem in zip(*gmms)]
        included = torch.stack(included)
        zh = self.model.merge_zh(zh, gmms, mask=~included)
        for z, path, include in zip(zh, paths, included):
            z = z[include]
            mesh = self.get_mesh(z, res)
            files_utils.export_mesh(mesh, path)

    def init_dict(self, paths, ids):
        values = []
        empty_path = paths[0]
        for i, id in enumerate(ids):
            if id == 2646:
                empty_path = paths[i]
        print(empty_path)
        values.append((self.load_file(empty_path, disclude=[i for i in range(16)])))
        # print("values[0] has ", values[0][3].shape, " gmms")
        ids = [0] + ids
        for i in paths:
            values.append((self.load_file(i)))
        self.loaded_files_dict = dict(zip(ids, values))
        print("loaded files dict length is:", len(self.loaded_files_dict))

    def get_indices(self):
        return list(self.loaded_files_dict.keys())

    def get_values(self):
        return list(self.loaded_files_dict.values())

    def get_alpha(self, id, bottom, mid, top):
        top_i, mid_i, bottom_i = self.loaded_files_indices_dict.get(id)
        # bottom_i = [1, 4, 9, 12]
        # mid_i = [0, 5, 7, 8, 13, 15]
        # top_i = [2, 6, 10, 14]
        # handles = [3, 11]
        alphas = [0.0 for i in range(16)]
        for i in bottom_i:
            alphas[i] = bottom / 100
        for i in mid_i:
            alphas[i] = mid / 100
        for i in top_i:
            alphas[i] = top / 100
        return alphas

    def __init__(self, opt: Options):
        self.opt = opt
        model: Tuple[OccGen, Options] = train_utils.model_lc(opt)
        self.model, self.opt = model
        self.model.eval()
        self.temperature = 1.
        self.plot_scale = self.get_plot_scale()
        self.mid: Optional[T] = None
        self.gmms: Optional[TN] = None
        self.get_rotation = utils.rotation_utils.rand_bounded_rotation_matrix(100000)
        # self.load_random("chairs_slides")
        self.meshing = mcubes_meshing.MarchingCubesMeshing(self.device, scale=self.plot_scale, max_num_faces=20000)
        self.loaded_files_dict = {}
        self.loaded_files_indices_dict = {
        2: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 3, 4, 9, 11, 12]),
        3: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 3, 4, 9, 11, 12]),
        1593: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 3, 4, 9, 11, 12]),
        1606: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 3, 4, 9, 11, 12]),
        1615: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 3, 4, 9, 11, 12]),
        31: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 4, 9, 12]),
        188: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 3, 4, 9, 11, 12]),
        209: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 4, 9, 12]),
        218: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 3, 4, 9, 11, 12]), #maybe remove 3, 11 for better legs
        262: ([2, 6, 7, 10, 14, 15], [4, 5, 12, 13], [0, 1, 3, 8, 9, 11]),
        314: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 4, 9, 12]),
        573: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 3, 4, 9, 11, 12]),
        725: ([2, 5, 6, 7, 10, 13, 14, 15], [0, 3, 8, 11], [1, 4, 9, 12]), #5, 3, could be in mid
        1406: ([6, 7, 14, 15], [0, 5, 8, 13], [1, 2, 3, 4, 9, 10, 11, 12]), #maybe remove 3, 11 for better legs
        1579: ([6, 7, 14, 15], [0, 2, 3, 5, 8, 10, 11, 13], [1, 4, 9, 12]), #2, 10 could be in top
        2646: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 4, 9, 12]),
        4442: ([2, 6, 7, 10, 14, 15], [0, 4, 5, 8, 12, 13], [1, 3, 9, 11]),
        5380: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 4, 9, 12]),
    }


def attention_to_image(attn: T):
    attn = attn.view(1, 1, *attn.shape)
    attn = nnf.interpolate(attn, scale_factor=16)[0, 0]
    image = image_utils.to_heatmap(attn) * 255
    image = image.numpy().astype(np.uint8)
    return image


def look_on_attn():
    attn = files_utils.load_pickle(f'../assets/checkpoints/occ_gmm_chairs_sym/attn/samples_5901')
    attn_b = attn["attn_b"]
    attn_a = attn["attn_a"]
    for j in range(1):
        attn_b_ = torch.stack(attn_b).max(-1)[0][:, 0]
        for i in range(4):
            image = attn_b_[i].float()
            image_ = image.gt(image.mean() + image.std()).float()
            for_print = (image_ - torch.eye(16)).relu().bool()
            inds = torch.where(for_print)
            print(inds[0].tolist())
            print(inds[1].tolist())
            image = attention_to_image(image)
            files_utils.save_image(image, f'../assets/checkpoints/occ_gmm_chairs_sym/attn/b_{i}_5901.png')
    return


def beautify():
    path = f"{constants.RAW_ROOT}/ui_export01"
    mesh = files_utils.load_mesh(path)
    mesh = mesh_utils.decimate_igl(mesh, 50000)
    mesh = mesh_utils.trimesh_smooth(mesh, iterations=40)
    files_utils.export_mesh(mesh, f"{constants.RAW_ROOT}smooth/ui_export01_smooth2")


# -------------------------------------

app = Flask(__name__)
CORS(app)
opt_ = Options(device=CUDA(0), tag='chairs_sym_hard', model_name='occ_gmm').load()
inference = Inference(opt_)
names = files_utils.collect(f"{opt_.cp_folder}/fresh_mix_origin/", '.pkl')
paths = [''.join(item) for item in names]  # [:2]
sample_names = [name[1] for name in names]
sample_numbers = [int(s.split("_")[1]) for s in sample_names]
inference.init_dict(paths, sample_numbers)

# temp_mesh = trimesh.load_mesh('model_edelman.obj')
# trimesh_scene = trimesh.Scene(geometry=temp_mesh)
# trimesh_scene.set_camera(distance=3, fov=[50, 50])
# trimesh_scene.save_image(resolution=(100,100))
for key, value in inference.loaded_files_dict.items():
    print(key)


def get_numbers():
    return inference.get_indices()


def get_prediction():
    return inference.plot_doriro_flask([5047], prefix="doriro", res=256)
    # return "fish"


def get_mix(chairs, alphas):
    return inference.mix_multiple_files_doriro_flask(chairs, alphas=alphas, res=256)
    # return "fish"


def get_mix_gltf(chairs, alphas):

    return inference.mix_multiple_files_doriro_flask_gltf(chairs, alphas=alphas, res=256)
    # return "fish"
def save_chair_png(mesh, path):
    # mesh = trimesh.load_mesh('model_edelman.obj')
    trimesh_scene = trimesh.Scene(geometry=mesh)

    # mesh_o3d = mesh.as_open3d
    # mesh_o3d.compute_vertex_normals()
    # material = open3d.visualization.rendering.Material()
    # render = open3d.visualization.rendering.OffscreenRenderer(640, 480)
    # render.scene.add_geometry("model", mesh_o3d, material)
    # render.scene.camera.look_at([0, 0, 0], [0, 10, 0], [0, 0, 1])
    # img_o3d = render.render_to_image()
    # success = open3d.io.write_image(path, img_o3d)


    trimesh_scene.add_geometry(geometry=mesh)
    angle = np.deg2rad(-135)
    direction = [0, 1, 0]
    center = [0, 0, 0]
    rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
    mesh.apply_transform(rot_matrix)
    angle = np.deg2rad(20)
    direction = [1, 0, 0]
    rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
    mesh.apply_transform(rot_matrix)
    mesh.apply_translation([100, -100, 0])
    trimesh_scene.set_camera(distance=3, fov=[50, 50])
    # window_conf = gl.Config(double_buffer=True, depth_size=24)
    data = trimesh_scene.save_image(resolution=(500,500))
    rendered = Image.open(trimesh.util.wrap_as_stream(data))
    rendered = rendered.convert('L')
    bw = rendered.point(lambda x: 0 if x < 200 else 255, '1')
    bw.save(path, format="png")
    return path

def get_mix_json(chairs, alphas):
    return inference.mix_multiple_files_doriro_flask_json(chairs, alphas=alphas, res=256)
    # return "fish"


@app.route('/predict')
def predict():
    result = get_prediction()
    return jsonify({'path': result})


@app.route('/mix', methods=['POST'])
def mix():
    global restart_counter
    global inference
    data = request.get_json()
    data = data.get('body', '')
    data = json.loads(data)
    name = data.get('name', '')
    print(name)
    chairs = [int(data.get('chair_1', '')), int(data.get('chair_2', '')), int(data.get('chair_3', '')),
              int(data.get('chair_4', ''))]
    alphas = [inference.get_alpha(chairs[0], int(data.get('chair_1_bottom', '')), int(data.get('chair_1_mid', '')),
                                  int(data.get('chair_1_top', ''))),
              inference.get_alpha(chairs[1], int(data.get('chair_2_bottom', '')), int(data.get('chair_2_mid', '')),
                                  int(data.get('chair_2_top', ''))),
              inference.get_alpha(chairs[2], int(data.get('chair_3_bottom', '')), int(data.get('chair_3_mid', '')),
                                  int(data.get('chair_3_top', ''))),
              inference.get_alpha(chairs[3], int(data.get('chair_4_bottom', '')), int(data.get('chair_4_mid', '')),
                                  int(data.get('chair_4_top', '')))]
    print(chairs)
    print(alphas)
    restart_counter += 1
    print('restart_counter value is: ',restart_counter)
    if(restart_counter == 3):
        inference = Inference(opt_)
        names = files_utils.collect(f"{opt_.cp_folder}/fresh_mix_origin/", '.pkl')
        paths = [''.join(item) for item in names]  # [:2]
        sample_names = [name[1] for name in names]
        sample_numbers = [int(s.split("_")[1]) for s in sample_names]
        inference.init_dict(paths, sample_numbers)
        restart_counter = 0
    return get_mix_gltf(chairs, alphas)

@app.route('/archive', methods=['POST'])
def archive():
    data = request.get_json()
    data = data.get('body', '')
    data = json.loads(data)
    name = data.get('name', '')
    creator = data.get('creator', '')
    print(name)
    chairs = [int(data.get('chair_1', '')), int(data.get('chair_2', '')), int(data.get('chair_3', '')),
              int(data.get('chair_4', ''))]
    alphas = [inference.get_alpha(chairs[0], int(data.get('chair_1_bottom', '')), int(data.get('chair_1_mid', '')),
                                  int(data.get('chair_1_top', ''))),
              inference.get_alpha(chairs[1], int(data.get('chair_2_bottom', '')), int(data.get('chair_2_mid', '')),
                                  int(data.get('chair_2_top', ''))),
              inference.get_alpha(chairs[2], int(data.get('chair_3_bottom', '')), int(data.get('chair_3_mid', '')),
                                  int(data.get('chair_3_top', ''))),
              inference.get_alpha(chairs[3], int(data.get('chair_4_bottom', '')), int(data.get('chair_4_mid', '')),
                                  int(data.get('chair_4_top', '')))]
    print(chairs)
    print(alphas)
    print("creating mesh and saving it")
    mesh = inference.mix_multiple_files_doriro_flask_archive(chairs, alphas)
    print("opening json and updating it")
    # open the json file, add another chair.
    file_name = f"{inference.opt.archive_json_folder}/ArchiveList.json"
    with open(file_name) as fp:
        listObj = json.load(fp)
        listObj.append({
            "itemName": f"{name}",
            "year": "2022",
            "creator": f"{creator} X chAIr",
            # "chair_1": chairs[0],
            # "chair_2": chairs[1],
            # "chair_3": chairs[2],
            # "chair_4": chairs[3],
            # "chair_1_values": [data.get('chair_1_top', ''), data.get('chair_1_mid', ''), data.get('chair_1_bottom', '')],
            # "chair_2_values": [data.get('chair_2_top', ''), data.get('chair_2_mid', ''), data.get('chair_2_bottom', '')],
            # "chair_3_values": [data.get('chair_3_top', ''), data.get('chair_3_mid', ''), data.get('chair_3_bottom', '')],
            # "chair_4_values": [data.get('chair_4_top', ''), data.get('chair_4_mid', ''), data.get('chair_4_bottom', '')],
        })
    with open(file_name, 'w') as json_file:
        json.dump(listObj, json_file,
                  indent=4,
                  separators=(',', ': '))
    print("done with json, saving file")
    gltf_to_save = trimesh.exchange.gltf.export_glb(mesh, include_normals=True)
    with open(f"{inference.opt.archive_model_folder}/{name}.gltf", 'wb') as f:
        f.write(gltf_to_save)
    return save_chair_png(mesh, path=f"{inference.opt.archive_image_folder}/chair_{name}.png")


@app.route('/gltf', methods=['POST'])
def gltf():
    # data = request.get_json()
    # data = data.get('body', '')
    # data = json.loads(data)
    # name = data.get('name', '')
    # print(name)
    # chairs = [int(data.get('chair_1', '')), int(data.get('chair_2', '')), int(data.get('chair_3', '')),
    #           int(data.get('chair_4', ''))]
    # alphas = [get_alpha(int(data.get('chair_1_bottom', '')), int(data.get('chair_1_mid', '')),
    #                     int(data.get('chair_1_top', ''))),
    #           get_alpha(int(data.get('chair_2_bottom', '')), int(data.get('chair_2_mid', '')),
    #                     int(data.get('chair_2_top', ''))),
    #           get_alpha(int(data.get('chair_3_bottom', '')), int(data.get('chair_3_mid', '')),
    #                     int(data.get('chair_3_top', ''))),
    #           get_alpha(int(data.get('chair_4_bottom', '')), int(data.get('chair_4_mid', '')),
    #                     int(data.get('chair_4_top', '')))]
    # print(chairs)
    # print(alphas)

    # return obj
    # mesh = trimesh.Trimesh(vertices=vs, faces=faces)
    # print("trying to convert obj to gltf using glb")
    # return trimesh.exchange.gltf.export_glb(mesh, include_normals=False)

    print("trying to import obj to trimesh")
    mesh = trimesh.load_mesh('model2.obj')
    print("trying to convert obj to gltf using glb")
    result2 = trimesh.exchange.gltf.export_glb(mesh, include_normals=True)
    return (result2)

    # return get_mix_gltf(chairs, alphas)
    # return jsonify({'path': 'model1.obj'})
    # result = get_mix_json(chairs, alphas)
    # with open('json_data.json', 'w') as outfile:
    #     json.dump(result, outfile)
    # return jsonify({'result': result})


@app.route('/nums')
def nums():
    result = get_numbers()
    return jsonify({'numbers installed': result})


@app.route('/values')
def values():
    result = inference.get_values()
    return jsonify({'values installed': result})


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.get('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


if __name__ == '__main__':
    app.run(threaded=False)
