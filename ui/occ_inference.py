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
import trimesh
import random
import json
import numpy as np
import io
from PIL import Image
import open3d
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

class Inference:


    def split_shape(self, mu: T) -> T:
        b, g, c = mu.shape
        # rotation_mat = mesh_utils.get_random_rotation(b).to(self.device)
        # mu_z: T = torch.einsum('bad,bgd->bga', rotation_mat, mu)[:, :, -1]
        mask = []
        for i in range(b):
            axis = torch.randint(low=0, high=c, size=(1,)).item()
            random_down_top_order = mu[i, :, axis].argsort(dim=-1, descending = torch.randint(low=0, high=2, size=(1,)).item() == 0)
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

                time_b = sdf_mesh.create_mesh_old(self.get_occ_fun(z, gmm), device=self.opt.device, scale=self.plot_scale, verbose=False,
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
                files_utils.export_mesh(mesh, f'{self.opt.cp_folder}/occ_{prefix}/{prefix}{name}')
                # files_utils.save_np(mesh[1], f'{self.opt.cp_folder}/vox/{prefix}{name}') #TODO Doriro - good export for interpolation?
                if gmms is not None:
                    files_utils.export_gmm(gmms, i, f'{self.opt.cp_folder}/gmms_{prefix}/{prefix}{name}')
            if verbose:
                print(f'done {i + 1:d}/{len(z):d}')

    def plot_occ_new_chairs(self, z: Union[T, TS], gmms: Optional[List[TS]], prefix: str, res=200, verbose=False,
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
                files_utils.export_mesh(mesh, f'{self.opt.cp_folder}/occ/{prefix}{name}', time=False)
                # files_utils.save_np(mesh[1], f'{self.opt.cp_folder}/vox/{prefix}{name}') #TODO Doriro - good export for interpolation?
                if gmms is not None:
                    files_utils.export_gmm(gmms, i, f'{self.opt.cp_folder}/gmms/{prefix}{name}', time=False)
            if verbose:
                print(f'done {i + 1:d}/{len(z):d}')

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
        zh_ = torch.cat(zh_, dim=0).unsqueeze(0).to(self.device)
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

    def get_mesh_interpolation(self, z: T, res: int, mask:TN, alpha: T) -> Optional[T_Mesh]:

        def forward(x: T) -> T:
            nonlocal z, alpha
            x = x.unsqueeze(0)
            out = self.model.occ_head(x, z, mask=mask, alpha=alpha)[0, :]
            out = 2 * out.sigmoid_() - 1
            return out

        mesh = self.meshing.occ_meshing(forward, res=res)
        return mesh

    def get_mesh_interpolation_multiple(self, z: T, res: int, mask:TN, alpha: T) -> Optional[T_Mesh]:

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
                mask[0, zh_a.shape[1]: ] = True
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
                mask[0, zh_a.shape[1]: ] = True
                zh_a = torch.cat((zh_a, padding_a), dim=1)
            if (zh_b.shape[1] < pad_length):
                padding_b = torch.zeros(1, abs(zh_b.shape[1] - pad_length), zh_a.shape[-1], device=zh_a.device)
                mask[1, zh_a.shape[1]: ] = True
                zh_b = torch.cat((zh_b, padding_b), dim=1)
            if (zh_c.shape[1] < pad_length):
                padding_c = torch.zeros(1, abs(zh_c.shape[1] - pad_length), zh_a.shape[-1], device=zh_a.device)
                mask[1, zh_a.shape[1]: ] = True
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
            files_utils.export_mesh_edelman(mesh, f"{self.opt.cp_folder}/{folder}/trial_alpha_edelman_{alpha_x}", time=False)
            # files_utils.export_list(colors.tolist(), f"{self.opt.cp_folder}/{folder}/trial_alpha_{alpha_x}_faces")
        logger.reset_iter()

    def interpolate_from_files_doriro_angles(self, item_a: Union[str, int], item_b: Union[str, int], alpha_x: float, res: int,
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
            files_utils.export_mesh_edelman(mesh, f"{self.opt.cp_folder}/{folder}/trial_alpha_edelman_{counter}", time=False)
            # files_utils.export_list(colors.tolist(), f"{self.opt.cp_folder}/{folder}/trial_alpha_{alpha_x}_faces")
        logger.reset_iter()

    def interpolate_from_files_doriro_three(self, item_a: Union[str, int], item_b: Union[str, int], item_c: Union[str, int], alpha_a: float, alpha_b: float, alpha_c: float, res: int,
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

    def interpolate_from_files_doriro_multiple_alphas(self, items: List[Union[str, int]], alphas: List[List[float]], res: int,
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
        fixed_items = torch.tensor(numbers) # TODO DORI
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
    def plot_doriro(self, chair_ids: List[int], prefix: Union[str, int] = "sample", verbose=False, interpolate: bool = False, res: int = 200, size: int = -1,
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
        self.plot_occ_new_chairs(zh, gmms, prefix, verbose=verbose, use_item_id=use_item_id,
                       res=res, fixed_items=names)

    @models_utils.torch_no_grad
    def plot_random_doriro(self, amount: int=1, prefix: Union[str, int] = "sample", verbose=False, res: int = 200,
             names: Optional[List[str]] = None, noise: float = 0):
        z = self.model.get_random_embeddings(amount, noise, only_noise=False)

        files_utils.save_pickle(z, f'{self.opt.cp_folder}/{"random_pickle_2"}')

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
        fixed_items = torch.tensor(numbers) # TODO DORI
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
    def random_samples(self,  nums_sample, res=256):
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
            for gmm in gmms}, #TODO doriro
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
        zh, gmms, _ , base_inds = self.load_file(path, disclude=z_idx)
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
    def mix_file_video(self, path: str, z_idx, replace_inds, num_samples: int, counter, res=220):
        '''
        Randomize a part of the file (inserted as a pkl).
        :param path:
        :param z_idx:
        :param replace_inds:
        :param num_samples:
        :param res:
        :return:
        '''
        zh, gmms, _ , base_inds = self.load_file(path, disclude=z_idx)
        folder = files_utils.split_path(item_a)[0].split('/')[-1]
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
            files_utils.export_mesh_edelman(mesh, f"{self.opt.cp_folder}/{folder}/trial_alpha_edelman_{counter + i}", time=False)
            # files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/occ_mix/mix_{i:02d}")
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
        zh, gmms, _ , base_inds = self.load_file(path_1, disclude=z_idx)
        print(path_1)
        print(path_2)
        print(path_3)
        zh_2, gmms_2, _2 , base_inds_2 = self.load_file(path_2)
        zh_3, gmms_3, _3 , base_inds_3 = self.load_file(path_3)
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
    def mix_multiple_files_doriro(self, paths: List[str],  alphas: List[List[float]], res=220, name = "mix"):
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
                    combined_added_gmm[k][:,:,i,:] = sum(alphas[j][i]*gmms_list[j][k][:,:,i,:] for j in range(len(gmms_list)))
                elif k == 1:
                    combined_added_gmm[k][:,:,i,:,:] = sum(alphas[j][i]*gmms_list[j][k][:,:,i,:,:] for j in range(len(gmms_list)))
                elif k == 2:
                    combined_added_gmm[k][:,:,i] = sum(alphas[j][i]*gmms_list[j][k][:,:,i] for j in range(len(gmms_list)))

        # gmms_new = [item[:, :, replace_inds_t] for item in gmms_new[0]]
        # gmms_new = [item[:, :, replace_inds_t] for item in gmms_2]
        # name = files_utils.get_time_name('light')

        gmms_new = combined_added_gmm
        # make zh something that makes sense.

        # option 1 - multiply each zh component with correct alpha and then combine them all.
        unsqueezed_zh = [zh[0].unsqueeze(0) for zh in zh_list]
        for i in range(16):
            unsqueezed_zh[0][:, i,:] = sum(alphas[j][i] * unsqueezed_zh[j][:, i,:] for j in range(len(unsqueezed_zh)))

        # option 2 - just make it 0.5 for all alphas.
        # option 3 - previous.
        # zh_ = torch.cat((empty_zh, *unsqueezed_zh), dim=1)
        zh_ = unsqueezed_zh[0]
        gmms_ = [torch.cat((item, item_new[0].unsqueeze(0)), dim=2) for item, item_new in zip(empty_gmms, gmms_new)]
        zh_ = self.model.merge_zh_step_a(zh_, [gmms_])
        zh_, _ = self.model.affine_transformer.forward_with_attention(zh_)
        mesh = self.get_mesh(zh_[0], res, None)
        # files_utils.export_gmm(gmms_, 0, f"{self.opt.cp_folder}/occ_mix/gmm_{name}", time=False)
        # self.save_light(self.opt.cp_folder, gmms_)
        # save_dict = {'ids': base_inds | {select_random[i].item(): replace_inds},
        #              'gmm': [item.cpu() for item in gmms_]}
        # path = f"{self.opt.cp_folder}/occ_mix_light/{name}_{i:02d}"
        # files_utils.save_pickle(save_dict, path)
        print(files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/occ_mix/mix_{name}"))

    def plot_only_parts(self, item, item_id, included_list, name = "only_selected_parts"):
        self.set_items((item_id))
        included_list_for_tensor = [[0, i] for i in included_list]
        included = torch.tensor(included_list_for_tensor, dtype=torch.int64)
        zh_, gmms_1, split, info = inference.load_file_doriro(item, included_list)

        # Extract a gmm, create a fake "included", and see if a mesh is available. Afterwards understand where to insert the alpha
        # variable to make combinations.
        mesh = inference.get_mesh_from_mid(gmms_1, included=included, res=200)
        print(files_utils.export_mesh(mesh, f'{inference.opt.cp_folder}/occ_bla/bla_{name}'))

    def random_gt(self):
        ds_train = mesh_datasets.CacheDataset(self.opt.dataset_name, self.opt.num_samples, self.opt.data_symmetric)
        ds_test = mesh_datasets.CacheDataset(self.opt.dataset_name.replace('train', 'test'), self.opt.num_samples, self.opt.data_symmetric)
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


def attention_to_image(attn: T):
    attn = attn.view(1, 1, *attn.shape)
    attn = nnf.interpolate(attn, scale_factor=16)[0,0]
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

# def send_mail(send_from, send_to, subject, message, files=[],
#               server="localhost", port=587, username='', password='',
#               use_tls=True):
def send_mail(attach_file_name='model.gltf'):
    """Compose and send email with provided info and attachments.
    """
    mail_content = '''Hello,
    This is a test mail.
    In this mail we are sending some attachments.
    The mail is sent using Python SMTP library.
    Thank You
    '''
    # The mail addresses and password
    sender_address = 'spaghetti.ai.system@gmail.com'
    sender_pass = 'spaghetti232'
    receiver_address = 'dori203@gmail.com'
    # Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'A test mail sent by Python. It has an attachment.'
    # The subject line
    # The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))
    # attach_file_name = 'TP_python_prev.pdf'
    attach_file = open(attach_file_name, 'rb')  # Open the file as binary mode
    payload = MIMEBase('application', 'octate-stream')
    payload.set_payload((attach_file).read())
    encoders.encode_base64(payload)  # encode the attachment
    # add payload header with filename
    payload.add_header('Content-Decomposition', 'attachment', filename=attach_file_name)
    message.attach(payload)
    # Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
    session.starttls()  # enable security
    session.login(sender_address, sender_pass)  # login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print('Mail Sent')

if __name__ == '__main__':
    from utils import image_utils
    # send_mail()
    # beautify()
    # look_on_attn()
    # opt_ = Options(device=CUDA(0), tag='tables_no_dis_split', model_name='occ_gmm').load()
    # TODO RUN THIS
    opt_ = Options(device=CUDA(0), tag='chairs_sym_hard', model_name='occ_gmm').load()
    inference = Inference(opt_)
    #

    # inference.load_random("chairs_slides")
    # TODO DORI
    # inference.random_samples(2,res=100)
    # inference.plot(prefix="sample", size=2, res=200, interpolate=False)
    # inference.plot_doriro([188])
    # TODO THE LINE BELOW WAS PREIOUSLY USED
    # inference.plot_doriro([31],prefix="doriro", res=400)
    # inference.plot_doriro([2,3,31,188,209,218,262,314,573,725,1406,1579,1587,1593,1606,1615,2646,4442,5380,6063,6129,6240,6567],prefix="doriroz", res=500)



    # inference.plot_mix()
    names = files_utils.collect(f"{opt_.cp_folder}/fresh_mix_origin/", '.pkl')
    # names = files_utils.collect(f"{opt_.cp_folder}/alpha_random_test/", '.pkl')
    paths = [''.join(item) for item in names] #[:2]
    sample_names = [name[1] for name in names]
    # sample_numbers = [int(s.split("_")[0]) for s in sample_names]
    sample_numbers = [int(s.split("_")[1]) for s in sample_names]
    num_to_path_dict = dict(zip(sample_numbers,paths))

    # item_a = num_to_path_dict.get(6756)
    # item_b = num_to_path_dict.get(6764)
    # item_a = num_to_path_dict.get(188)
    # item_b = num_to_path_dict.get(314)
    item_a = num_to_path_dict.get(31)
    item_b = num_to_path_dict.get(209)
    item_c = num_to_path_dict.get(1606)
    item_d = num_to_path_dict.get(725)
    item_e = num_to_path_dict.get(218)


    # item_c = num_to_path_dict.get(6240)
    # item_d = num_to_path_dict.get(188)
    # inference.plot_random_doriro(10, res=256)
    # inference.plot_random_doriro(5, noise=0.05, res=256)
    print("fish")
    # inference.plot_random_doriro(5, noise=0.15, res=256)
    print("fish")
    # inference.plot_random_doriro(5, noise=0.18, res=256)
    # inference.plot_random_doriro(5, noise=0.25, res=256)
    print("fish")
    # inference.plot_random_doriro(10, noise=1.3, res=256)
    print("fish")
    # inference.plot_random_doriro(5, noise=1.0, res=256)
    # inference.mix_file(item_a,
    #                    (2, 6, 7, 10, 14, 15), (2, 6, 7, 10, 14, 15), num_samples=10, res=256)
    # inference.mix_file(item_a,
    #                    (1,3,4,9,11,12), (1,3,4,9,11,12), num_samples=10, res=256)
    items = [item_a, item_b]
    # alpha_b = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]
    # alpha_c = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    # left_feet = [9,12]
    # right_feet = [1,4]
    # handles = [3,11]
    # seat = [0,5,8,13]
    # mid_seat = [2,10]
    # top_back_seat = [14, 6]
    # sides_seat = [15,7]
    half = [0.5 for i in range(16)]
    # alpha_a = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0]
    # alpha_b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0]
    # alpha_c = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0]
    # alpha_d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0]
    # TODO update num accordingly
    # num = 31
    # loaded_files_indices_dict = {
    #     31: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 4, 9, 12]),
    #     188: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 3, 4, 9, 11, 12]),
    #     209: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 4, 9, 12]),
    #     218: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 3, 4, 9, 11, 12]), #maybe remove 3, 11 for better legs
    #     262: ([2, 6, 7, 10, 14, 15], [4, 5, 12, 13], [0, 1, 3, 8, 9, 11]),
    #     314: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 4, 9, 12]),
    #     573: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 3, 4, 9, 11, 12]),
    #     725: ([2, 5, 6, 7, 10, 13, 14, 15], [0, 3, 8, 11], [1, 4, 9, 12]), #5, 3, could be in mid
    #     1406: ([6, 7, 14, 15], [0, 5, 8, 13], [1, 2, 3, 4, 9, 10, 11, 12]), #maybe remove 3, 11 for better legs
    #     1579: ([6, 7, 14, 15], [0, 2, 3, 5, 8, 10, 11, 13], [1, 4, 9, 12]), #2, 10 could be in top
    #     2646: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 4, 9, 12]),
    #     4442: ([2, 6, 7, 10, 14, 15], [0, 4, 5, 8, 12, 13], [1, 3, 9, 11]),
    #     5380: ([2, 6, 7, 10, 14, 15], [0, 5, 8, 13], [1, 4, 9, 12]),
    # }
    # item_a = num_to_path_dict.get(num)
    # alpha_top = [0.0 for i in range(16)]
    # alpha_mid = [0.0 for i in range(16)]
    # alpha_bottom = [0.0 for i in range(16)]
    # alpha_all = [0.0 for i in range(16)]
    # top_i, mid_i, bottom_i = loaded_files_indices_dict.get(num)

    # for i in top_i:
    #     alpha_top[i] = 1.0
    #     alpha_all[i] = 1.0
    # for i in mid_i:
    #     alpha_mid[i] = 1.0
    #     alpha_all[i] = 1.0
    # for i in bottom_i:
    #     alpha_bottom[i] = 1.0
    #     alpha_all[i] = 1.0

    # inference.plot_random_doriro(4, noise=1.3, res=200)
    # inference.mix_file(item_a,
    #                    (2, 6, 7, 10, 14, 15), (2, 6, 7, 10, 14, 15), num_samples=2, res=100)
    # items = [item_a]

    # inference.mix_multiple_files_doriro([item_a, item_b],[half,half],name="top")
    # inference.mix_multiple_files_doriro(items,[alpha_mid],name="mid")
    # inference.mix_multiple_files_doriro(items,[alpha_bottom],name="bottom")
    # inference.mix_multiple_files_doriro(items,[alpha_all],name="all")

    # inference.plot_only_parts(item_a, num, top_i, name="top")
    # inference.plot_only_parts(item_a, num, mid_i, name="mid")
    # inference.plot_only_parts(item_a, num, bottom_i, name="bottom")
    # inference.plot_only_parts(item_a, num, [i for i in range(16)], name="all")
    # print(bottom_i)
    # inference.plot_only_parts(item_a, num, [3,11], name="3_11")

    # inference.mix_2_files_doriro(item_b, item_a, item_d,
    #                    (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15), (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14), num_samples=1, res=200)
    # included_list = [0,1,2,3,4,5,6,78,9,10,11,12,13,14,15]
    # included_list = [0,1,2]
    # inference.plot_only_parts(item_b, 755, included_list)
    # inference.plot_only_parts(item_a, 2646, included_list)
    # inference.plot_only_parts(item_d, 2954, included_list)
    # for j in range(5):
    #     chair_ids = [random.randint(0,22) for i in range(4)]
    #     chairs = [names[i] for i in chair_ids]
    #     alphas = [random.random() for i in range(4)]
    #     a = 1/sum(alphas)
    #     alphas = [alpha * a for alpha in alphas]
        # print(sum(alphas))
        # print(chair_ids)
        # inference.interpolate_from_files_doriro_multiple(items=chairs, alphas=alphas, res=256)
        # inference.plot_doriro(chair_ids,prefix="doriro", res=256)
    # items = (2646)
    # items = torch.tensor(items, dtype=torch.int64, device=inference.device)
    # inference.set_items((2646,755,188,2954))
    # zh, z, gmms = inference.set_items_get_all(items)
    # gmm_mask = torch.zeros(gmms[0].shape[2], dtype=torch.bool)

    # included_list_for_tensor = [[0,i] for i in included_list]
    # included = torch.tensor(included_list_for_tensor, dtype=torch.int64)

    # combined
    # included_list_combined = [[0,1],[1,10]]
    # included_combined = torch.tensor(included_list_combined, dtype=torch.int64)
    #
    # zh_, gmms_1, split, info = inference.load_file_doriro(item_a, included_list=[1])
    # zh_, gmms_2, split, info = inference.load_file_doriro(item_c, included_list=[10])
    # united = torch.cat((gmms_1, gmms_2), dim=1)
    # mesh = inference.get_mesh_from_mid(united, included=included_combined, res=200)
    # zh_, gmms_1, split, info = inference.load_file_doriro(item_a, included_list)
    # zh_, gmms_2, split, info = inference.load_file_doriro(item_c, included_list)

    # print("trying to import obj to trimesh")
    # mesh = trimesh.load_mesh('model_edelman.obj')
    # mesh_o3d = mesh.as_open3d
    # mesh_o3d.compute_vertex_normals()
    # material = open3d.visualization.rendering.Material()

    # open3d.geometry.TriangleMesh.
    # mesh = open3d.geometry.TriangleMesh(mesh.as_open3d())
    # open3d.visualization.draw_geometries([mesh.as_open3d])
    # mesh_geo = mesh_o3d.scale(2,[0,0,0])
    # render = open3d.visualization.rendering.OffscreenRenderer(640, 480)
    # render.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 100000)
    # render.scene.enable_sun_light(True)

    # render.scene.set_background([0, 0, 0, 0])
    # render.scene.add_geometry("model", mesh_o3d, material)
    # render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))


    # render.scene.camera.look_at([0, 0, 0], [0, 10, 0], [0, 0, 1])
    # img_o3d = render.render_to_image()
    # success = open3d.io.write_image("mtest2.png", img_o3d)
    # if success:
    #     print("yeah")
    # render.scene.remove_geometry("model")
    # render.destroy()

    # mesh = open3d.io.read_triangle_mesh(armadillo_mesh.path)

    # TODO previous attempt
    # trimesh_scene = trimesh.Scene(geometry=mesh)
    # angle = np.deg2rad(-135)
    # direction = [0, 1, 0]
    # center = [0, 0, 0]
    # rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
    # mesh.apply_transform(rot_matrix)
    # angle = np.deg2rad(20)
    # direction = [1, 0, 0]
    # rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
    # mesh.apply_transform(rot_matrix)
    # mesh.apply_translation([100, -100, 0])
    # trimesh_scene.set_camera(distance=3, fov=[50, 50])
    # # trimesh_scene.show(resolution=[500, 500])
    # data = trimesh_scene.save_image(resolution=(500,500))
    # rendered = Image.open(trimesh.util.wrap_as_stream(data))
    # rendered = rendered.convert('L')
    # bw = rendered.point(lambda x: 0 if x < 200 else 255, '1')
    # bw.save("converted.png", format="png")

    # image = np.array(Image.open(io.BytesIO(data)))

    # files_utils.save_image(data, f'../assets/checkpoints/occ_gmm_chairs_sym/images/b_5901.png')

    #
    # print("trying to convert obj to gltf")
    # result = trimesh.exchange.gltf.export_gltf(mesh, include_normals=False, merge_buffers=True)
    #
    # print("trying to convert obj to gltf using glb")
    # result2 = trimesh.exchange.gltf.export_glb(mesh, include_normals=True)
    #
    # print("view results")
    #
    # with open("test_gltf_json.gltf", 'w') as f:
    #     json.dump(result, f, indent=4)
    #
    # trimesh_scene = trimesh.Scene(geometry=mesh)
    # with open("test_glb.gltf", 'wb') as f:
    #     f.write(result2)

    # export = trimesh_scene.export(file_type='gltf')
    # for file_name, data in result.items():
    #     with open(file_name, 'wb') as f:
    #         f.write(data)
    # print("fish")
    # print("reloading")
    # reloaded = trimesh.exchange.load.load_kwargs(
    #     trimesh.exchange.gltf.load_gltf(
    #         file_obj=None,
    #         resolver=trimesh.visual.resolvers.ZipResolver(export)))


    # included = torch.tensor([[0,1]], dtype=torch.int64)
    # Extract a gmm, create a fake "included", and see if a mesh is available. Afterwards understand where to insert the alpha
    # variable to make combinations.
    # mesh = inference.get_mesh_from_mid(gmms_1, included=included, res=200)
    # files_utils.export_mesh(mesh, f'{inference.opt.cp_folder}/occ_bla/bla')
    # mesh = inference.get_mesh_from_mid(gmms[0], included=included, res=200)
    print("set items successfully")
    # inference.interpolate_from_files_doriro_multiple(items=[item_a, item_b,item_c, item_d], alphas=[0.25, 0.25,0.25,0.25], res=256)
    # inference.interpolate(item_a=item_a, item_b=item_b, num_mid=3, res=100)
    # angles = [1/x for x in range(100,1)]
    # TODO angles work for video.

    # ----------------------------------------------------------------------

    #
    angles = np.arange(0.0, 1.0, 1.0/99)
    angles = angles.tolist()
    angles.append(1.0)
    print(len(angles), angles)
    frame_counter = 0
    for angle in angles:
        frame_counter += 1
        inference.interpolate_from_files_doriro_angles(item_a=item_a, item_b=item_b, alpha_x=angle, res=256, counter = frame_counter)
    print("finished a -> b, frame counter is: ", frame_counter)
    random_frames = 1
    inference.mix_file_video(item_b,
                      (2, 6, 7, 10, 14, 15), (2, 6, 7, 10, 14, 15), num_samples=random_frames, counter=frame_counter+1, res=256)
    frame_counter += random_frames
    print("finished random b, frame counter is: ", frame_counter)
    for angle in angles:
        frame_counter += 1
        inference.interpolate_from_files_doriro_angles(item_a=item_b, item_b=item_c, alpha_x=angle, res=256, counter = frame_counter)
    print("finished b -> c, frame counter is: ", frame_counter)

    for angle in angles:
        frame_counter += 1
        inference.interpolate_from_files_doriro_angles(item_a=item_c, item_b=item_d, alpha_x=angle, res=256, counter = frame_counter)
    print("finished c -> d, frame counter is: ", frame_counter)

    random_frames = 2
    inference.mix_file_video(item_d,
                       (2, 6, 7, 10, 14, 15), (2, 6, 7, 10, 14, 15), num_samples=random_frames, counter=frame_counter+1, res=256)
    frame_counter += random_frames
    print("finished random d, frame counter is: ", frame_counter)

    for angle in angles:
        frame_counter += 1
        inference.interpolate_from_files_doriro_angles(item_a=item_d, item_b=item_e, alpha_x=angle, res=256, counter = frame_counter)

    print("finished d -> e, frame counter is: ", frame_counter)

    for angle in angles:
        frame_counter += 1
        inference.interpolate_from_files_doriro_angles(item_a=item_e, item_b=item_a, alpha_x=angle, res=256, counter = frame_counter)

    print("finished d -> e, frame counter is: ", frame_counter)


    # ----------------------------------------------------------------------

    # inference.interpolate_doriro(4442, 262, num_mid=3, res=200)
    # inference.load_random("model") # TODO DORI
    # inference.plot()
    # inference.random_stash(1000, f"chairs_slides")
    # inference.random_gt()
    # names= ['cace287f0d784f1be6fe3612af521500', 'c953d7b4f0189fe6a5838970f9c2180d',
    #         'ca01fd0de2534323c594a0e804f37c1a', 'cb1986dd3e968310664b3b9b23ddfcbc']
    # names = ['cb714c518e3607b8ed4feff97703cf03', 'c9d5ff677600b0a1a01ae992b62200ab', 'c93a696c4b6f843122963ea1e168015d',
    #          'cbbbb3aebaf2e112ca07b3f65fc99919', 'ca2294ffc664a55afab1bffbdecd7709',
    #          'cb17f1916ade6e005d5f1108744f02f1', 'c88eb06478809180f7628281ecb18112', 'c93113c742415c76cffd61677456447e',
    #          'c976cb3eac6a89d9a0aa42b42238537d', 'c9fa3d209a43e7fd38b39a90ee80e328', 'ca84b42ab1cfc37be25dfc1bbeae5325',
    #          'ca9f1525342549878ad57b51c4441549', 'c833ef6f882a4b2a14038d588fd1342f', 'c92721a95fe44b018039b09dacd0f1a7',
    #          'cb867c64ea2ecb408043364ed41c1a79', 'cbbf0aacb76a1ed17b20cb946bceb58f', 'c9f5c127b44d0538cb340854b82a069f',
    #          'c98e1a3e61caec6a67d783b4714d4324', 'c95e8fc2cf96b9349829306a513f9466', 'c98c12e85a3f70a28ddc51277f2e9733']

    # names = ['cf911e274fb613fbbf3143b1cb6076a', 'cf93f33b52900c64bbf3143b1cb6076a', 'cfaff76a1503d4f562b600da24e0965', 'cfb555a4d82a600aca8607f540cc62ba', 'cfd42bf49322e79d8deb28944f3f72ef', 'd01da87ecbc2deea27e33924ab17ba05', 'd0e517321cdcd165939e75b12f2e5480', 'd0ee4253d406b3f05e9e2656aff7dd5b', 'd1a887a47991d1b3bc0909d98a1ff2b4', 'd1b28579fde95f19e1873a3963e0d14', 'd1cdd239dcbfd018bbf3143b1cb6076a', 'd1d308692cb4b6059a6e43b878d5b335', 'd1df81e184c71e0f26360e1e29a956c7']
    # names = ['d0e517321cdcd165939e75b12f2e5480']
    # names = ['5710'] TODO DORI
    # names = files_utils.collect(f"{opt_.cp_folder}/occ_mix_light_legs/", '.pkl')
    # names = [''.join(item) for item in names][:2]
    # inference.interpolate_seq(15, *names, res=256)
    # inference.mix_file(f"{opt_.cp_folder}/occ_mix_light_seat/light_01_25-18_00_30",
    #                    (2, 6, 7, 10, 14, 15), (2, 6, 7, 10, 14, 15), num_samples=30, res=220)
    # inference.plot_folder(f"{opt_.cp_folder}/occ_mix_light/", res=220)
    # inference.plot_folder(f"{constants.DATA_ROOT}ui_export/occ_gmm_tables_no_dis_split_01_27-22_45", res=220)
    # inference.plot_folder(f"{constants.DATA_ROOT}ui_export/occ_gmm_chairs_sym_hard_03_14-16_14", res=50)

    # inference.plot("mix", verbose=True, interpolate=False, res=10, size=50, names=names) TODO DORI
# "\begin{figure}
# \centering
# % \footnotesize
#     \begin{overpic}[width=1\columnwidth,tics=10, trim=0mm 0 0mm 0,clip]{figures/coarse_diagram_v2_hor-01.png}
#
#      \put(-.8,14){$\mathbf{\za}$}
#
#     \put(34.6,13.4){$\mathbf{\zB}$}
#
#      \put(67.9,13.4){$\mathbf{\zC}$}
#
#      \put(18.1,-2.2){$\netA$}
#      \put(52,-2.2){$\netB$}
#      \put(84.5,-2.2){$\netC$}
#
#     %  \put(-2,-2){{$\mathbf{\za}$: shape embedding}}
#     %  \put(33,-2){{$\mathbf{\zB}$: part embedding}}
#     %  \put(66,-2){{$\mathbf{\zC}$: contextual embedding}}
#
#     \end{overpic}
#
# \caption{Method overview. Our implicit shape generative model learns to decompose a shape embedding ${\za}$ into distinct embeddings $\zB$ that correspond to distinct 3D parts. Then, a mixing network outputs contextual embeddings $\zC$. Finally the implicit shape is given by a third occupancy network $\netC$ that is conditioned on the contextual part embeddings.}
#
#
# \label{fig:overview}
#
#
# \end{figure}
# "
