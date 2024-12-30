import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import trimesh
import numpy as np
from PIL import Image
import sys
import os
import logging
from typing import List, Optional, Tuple, Union
from flask import Response

# ------------------------------------- occ inference imports
sys.path.insert(0, 'C:/Users/dori2/Desktop/Bezalel/Year 5/pgmr/spaghetti_code/spaghetti_code')
import utils.rotation_utils
from custom_types import *
import constants
from data_loaders import mesh_datasets
from options import Options
from utils import train_utils, mcubes_meshing, files_utils, mesh_utils
from models.occ_gmm import OccGen
from models import models_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check GPU availability
def check_gpu_status():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        logger.info(f"GPU is available: {gpu_name}")
        logger.info(f"Number of GPUs: {gpu_count}")
        return True
    else:
        logger.warning("No GPU detected - running on CPU")
        return False

# Global variables
restart_counter = 0


class Inference:

    def load_file(self, info_path, included_list=None, disclude=None):
        """Unified method to load file data with optional filtering"""
        info = files_utils.load_pickle(''.join(info_path))
        keys = list(info['ids'].keys())
        items = torch.tensor([int(x.split('_')[1]) if type(x) is str else x for x in keys], 
                        dtype=torch.int64, device=self.device)
        zh, _, gmms_sanity, _ = self.model.get_embeddings(items)
        gmms = [item for item in info['gmm']]
        
        # Create mask for included/discluded gaussians
        gmm_mask = torch.ones(gmms[0].shape[2], dtype=torch.bool)
        if included_list is not None:
            gmm_mask.fill_(False)
            for i in included_list:
                gmm_mask[i] = True
        if disclude is not None:
            for i in disclude:
                gmm_mask[i] = False
                
        # Process gaussian indices
        zh_ = []
        split = []
        counter = 0
        for i, key in enumerate(keys):
            gaussian_inds = info['ids'][key]
            if disclude is not None:
                gaussian_inds = [ind for ind in gaussian_inds if ind not in disclude]
                info['ids'][key] = gaussian_inds
            gaussian_inds = torch.tensor(gaussian_inds, dtype=torch.int64)
            zh_.append(zh[i, gaussian_inds])
            split.append(len(split) + torch.ones(len(gaussian_inds), dtype=torch.int64, device=self.device))
            
        zh_ = torch.cat(zh_, dim=0).unsqueeze(0).to(self.device)
        gmms = [item[:, :, gmm_mask].to(self.device) for item in gmms]
        return zh_, gmms, split, info['ids']

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
        # print("A. Entering get_occ_fun")
        print(f"B. z shape: {z.shape}, device: {z.device}")
        print(f"C. gmm: {type(gmm) if gmm is not None else 'None'}")

        def forward(x: T) -> T:
            nonlocal z
            # print("D. Forward called with x shape:", x.shape)
            x = x.unsqueeze(0)
            # print("E. After unsqueeze x shape:", x.shape)

            # print("F. About to call occ_head")
            out = self.model.occ_head(x, z, gmm)[0, :]
            # print("G. occ_head output shape:", out.shape)

            if self.opt.loss_func == LossType.CROSS:
                print("H. Using CROSS loss")
                out = out.softmax(-1)
                out = -1 * out[:, 0] + out[:, 2]
            elif self.opt.loss_func == LossType.IN_OUT:
                # print("I. Using IN_OUT loss")
                print(f"Pre-sigmoid values - min: {out.min()}, max: {out.max()}, mean: {out.mean()}")
                # sig = out.sigmoid_()
                # print(f"Sig values - min: {sig.min()}, max: {sig.max()}, mean: {sig.mean()}")
                out = 2 * out.sigmoid_() - 1
                print(f"Final scaled values - min: {out.min()}, max: {out.max()}, mean: {out.mean()}")
            else:
                print("J. Using default loss")
                out.clamp_(-.2, .2)

            # print("K. Final output shape:", out.shape)
            return out

        if z.dim() == 2:
            print("L. Expanding z dimension")
            z = z.unsqueeze(0)
        print("M. Final z shape:", z.shape)
        return forward

    def get_mesh(self, z: T, res: int, gmm: Optional[TS], get_time=False) -> Optional[T_Mesh]:
        try:
            print("Environment info:")
            print(f"Python version: {sys.version}")
            print(f"PyTorch version: {torch.__version__}")
            print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
            print(f"Current working directory: {os.getcwd()}")

            print("1. Entering get_mesh")
            print(f"2. z device: {z.device}, shape: {z.shape}")
            print(f"3. Resolution: {res}")
            print(f"4. GMM: {type(gmm) if gmm is not None else 'None'}")

            with torch.no_grad():
                if get_time:
                    print("5. Time measurement branch")
                    time_a = self.meshing.occ_meshing(self.get_occ_fun(z, gmm), res=res, get_time=get_time)
                    print("6. First occ_meshing completed")

                    time_b = sdf_mesh.create_mesh_old(self.get_occ_fun(z, gmm), device=self.opt.device,
                                                      scale=self.plot_scale, verbose=False,
                                                      res=res, get_time=get_time)
                    print("7. create_mesh_old completed")
                    return time_a, time_b
                else:
                    print("8. Regular mesh generation branch")
                    print("9. About to call get_occ_fun")
                    occ_fun = self.get_occ_fun(z, gmm)
                    print("10. get_occ_fun completed")

                    print("11. About to call occ_meshing")
                    mesh = self.meshing.occ_meshing(occ_fun, res=res)
                    print(f"12. occ_meshing completed, mesh type: {type(mesh)}")

                    return mesh

        except Exception as e:
            print(f"Error in get_mesh: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None

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

    @models_utils.torch_no_grad
    def get_z_from_file(self, info_path):
        zh_, gmms, split, _ = self.load_file(info_path)
        zh_ = self.model.merge_zh_step_a(zh_, [gmms])
        zh, _ = self.model.affine_transformer.forward_with_attention(zh_)

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
    def combine_and_pad(zh_list: List[T]) -> Tuple[T, TN]:
        """Combines multiple tensors with padding to match the largest tensor.
        
        Args:
            zh_list: List of tensors to combine
            
        Returns:
            Tuple containing:
            - Combined tensor
            - Mask indicating padded values (None if no padding needed)
        """
        zh_length = len(zh_list)
        pad_length = max(zh.shape[1] for zh in zh_list)
        
        # If all tensors are same length, no padding needed
        if pad_length == min(zh.shape[1] for zh in zh_list):
            return torch.cat(zh_list, dim=0), None
            
        # Create mask and pad tensors
        mask = torch.zeros(zh_length, pad_length, device=zh_list[0].device, dtype=torch.bool)
        for i, zh in enumerate(zh_list):
            if zh.shape[1] < pad_length:
                padding = torch.zeros(1, pad_length - zh.shape[1], zh.shape[-1], device=zh.device)
                mask[i, zh.shape[1]:] = True
                zh_list[i] = torch.cat((zh, padding), dim=1)
                
        return torch.cat(zh_list, dim=0), mask

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

    @models_utils.torch_no_grad
    def interpolate_from_files(self, 
                            items: List[Union[str, int]], 
                            alphas: Optional[List[Union[float, List[float]]]] = None,
                            num_steps: int = 1,
                            res: int = 200,
                            export_colors: bool = False,
                            logger: Optional[train_utils.Logger] = None) -> Optional[T_Mesh]:
        """Interpolate between multiple chair designs.
        
        Args:
            items: List of file paths or item IDs to interpolate between
            alphas: Optional list of alpha values or weights for each item.
                    If None and len(items)==2, will generate num_steps interpolation steps
            num_steps: Number of interpolation steps (only used for 2 items without alphas)
            res: Resolution of output mesh
            export_colors: Whether to export attention face colors
            logger: Optional logger for progress tracking
            
        Returns:
            The generated mesh, or None if generation failed
        """
        # Handle simple interpolation case
        if len(items) == 2 and alphas is None:
            zh_a, zh_a_raw, _, _ = self.get_z_from_file(items[0])
            zh_b, zh_b_raw, _, _ = self.get_z_from_file(items[1])
            fixed_z = self.get_intersection_z(zh_a_raw, zh_b_raw)
            zh, mask = self.combine_and_pad([zh_a, zh_b])
            
            results = []
            if logger is None:
                logger = train_utils.Logger().start(num_steps)
                
            for i, alpha_ in enumerate(torch.linspace(0, 1, num_steps)):
                alpha = torch.tensor([1. - alpha_, alpha_], device=self.device)
                mesh = self.get_mesh_interpolation(zh, res, mask, alpha)
                
                if export_colors:
                    colors = self.get_attention_faces(mesh, zh, mask, fixed_z, alpha)
                    colors = (~colors).long()
                    files_utils.export_list(colors.tolist(), 
                        f"{self.opt.cp_folder}/interpolation/colors_{i:03d}")
                    
                results.append(mesh)
                logger.reset_iter()
                
            return results[0] if num_steps == 1 else results

        # Handle multiple item interpolation
        zh_list = [self.get_z_from_file(item)[0] for item in items]
        zh, mask = self.combine_and_pad(zh_list)
        
        if logger is None:
            logger = train_utils.Logger().start(1)
            
        # Convert alphas to tensor
        if alphas is None:
            alpha = torch.ones(len(items), device=self.device) / len(items)
        else:
            alpha = torch.tensor(alphas, device=self.device)
            
        mesh = self.get_mesh_interpolation_multiple(zh, res, mask, alpha)
        logger.reset_iter()
        
        return mesh

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
    def plot_chairs(self, 
                    chair_ids: Optional[List[int]] = None,
                    prefix: Union[str, int] = "sample",
                    verbose: bool = False,
                    res: int = 200,
                    random_amount: int = 0,
                    noise: float = 0,
                    return_result: bool = False) -> Optional[str]:
        """Unified method for plotting chairs.
        
        Args:
            chair_ids: List of chair IDs to plot. If None and random_amount=0, uses default test chairs
            prefix: Prefix for output files
            verbose: Whether to print verbose output
            res: Resolution of output mesh
            random_amount: If >0, generates this many random chairs instead of using chair_ids
            noise: Amount of noise to add when generating random chairs
            return_result: Whether to return the result path (for Flask endpoints)
        
        Returns:
            Path to generated file if return_result=True, otherwise None
        """
        if type(prefix) is int:
            prefix = f'{prefix:04d}'
            
        if random_amount > 0:
            # Generate random chairs
            z = self.model.get_random_embeddings(random_amount, noise, only_noise=True)
            zh, gmms, _ = self.model.occ_former(z)
            zh, _ = self.model.merge_zh(zh, gmms)
            names = None
            use_item_id = False
        else:
            # Use provided or default chair IDs
            numbers = chair_ids if chair_ids else [755, 2646]  # Default test chairs
            fixed_items = torch.tensor(numbers)
            names = [f'{i:02d}_' for i in fixed_items]
            zh, _, gmms, attn_a = self.model.get_embeddings(fixed_items.to(self.device))
            zh, attn_b = self.model.merge_zh(zh, gmms)
            use_item_id = True

        result = self.plot_occ(zh, gmms, prefix, verbose=verbose, 
                            use_item_id=use_item_id, res=res, 
                            fixed_items=names)
                            
        return result if return_result else None

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

    def plot_only_parts(self, item, item_id, included_list):
        self.set_items((item_id))
        included_list_for_tensor = [[0, i] for i in included_list]
        included = torch.tensor(included_list_for_tensor, dtype=torch.int64)
        zh_, gmms_1, split, info = inference.load_file(item, included_list)

        # Extract a gmm, create a fake "included", and see if a mesh is available. Afterwards understand where to insert the alpha
        # variable to make combinations.
        mesh = inference.get_mesh_from_mid(gmms_1, included=included, res=200)
        files_utils.export_mesh(mesh, f'{inference.opt.cp_folder}/occ_bla/bla')

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
        alphas = torch.zeros(16, dtype=torch.float32)
        alphas[torch.tensor(bottom_i)] = bottom / 100
        alphas[torch.tensor(mid_i)] = mid / 100 
        alphas[torch.tensor(top_i)] = top / 100
        return alphas.tolist()

    def mix_file_video_gltf(self, chair_id, replace_inds):
        zh, gmms, _, base_inds = self.loaded_files_dict.get(chair_id)
        replace_inds_t = torch.tensor(replace_inds, dtype=torch.int64, device=self.device)
        select_random = torch.rand(1000)
        select_random = select_random.argsort()
        select_random = select_random[:num_samples]

        zh_new, _, gmms_new, _ = self.model.get_embeddings(select_random.to(self.device))
        zh_new = zh_new[:, replace_inds_t]
        gmms_new = [item[:, :, replace_inds_t] for item in gmms_new[0]]
        zh_ = torch.cat((zh, zh_new[i].unsqueeze(0)), dim=1)
        gmms_ = [torch.cat((item, item_new[i].unsqueeze(0)), dim=2) for item, item_new in zip(gmms, gmms_new)]
        zh_ = self.model.merge_zh_step_a(zh_, [gmms_])
        zh_, _ = self.model.affine_transformer.forward_with_attention(zh_)
        mesh = self.get_mesh(zh_[0], res, None)
        files_utils.export_mesh_edelman(mesh, f"{self.opt.cp_folder}/{folder}/trial_alpha_edelman_{counter + i}",
                                        time=False)

################################################################

    @models_utils.torch_no_grad
    def mix_files(self, indices: List[int], alphas: List[List[float]], res=220, return_format='mesh'):
        """Unified method to mix multiple files with different return formats"""
        # Get base empty model
        empty_zh, empty_gmms, _, _ = self.loaded_files_dict.get(0)
        
        # Load all input files
        zh_list, gmms_list = [], []
        for i in indices:
            zh, gmms, _, _ = self.loaded_files_dict.get(i)
            zh_list.append(zh)
            gmms_list.append(gmms)

        # Mix GMMs
        combined_gmm = gmms_list[0]
        for i in range(16):
            for k in range(4):
                if k in (0, 3):
                    combined_gmm[k][:, :, i, :] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i, :] for j in range(len(gmms_list)))
                elif k == 1:
                    combined_gmm[k][:, :, i, :, :] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i, :, :] for j in range(len(gmms_list)))
                elif k == 2:
                    combined_gmm[k][:, :, i] = sum(
                        alphas[j][i] * gmms_list[j][k][:, :, i] for j in range(len(gmms_list)))

        # Mix latent vectors
        unsqueezed_zh = [zh[0].unsqueeze(0) for zh in zh_list]
        for i in range(16):
            unsqueezed_zh[0][:, i, :] = sum(
                alphas[j][i] * unsqueezed_zh[j][:, i, :] for j in range(len(unsqueezed_zh)))

        # Generate final mesh
        zh_ = unsqueezed_zh[0]
        gmms_ = [torch.cat((item, item_new[0].unsqueeze(0)), dim=2) 
                 for item, item_new in zip(empty_gmms, combined_gmm)]
        zh_ = self.model.merge_zh_step_a(zh_, [gmms_])
        zh_, _ = self.model.affine_transformer.forward_with_attention(zh_)
        mesh = self.get_mesh(zh_[0], res, None)
        
        # Return in requested format
        if return_format == 'mesh':
            return mesh
        elif return_format == 'gltf':
            vs, faces = mesh
            mesh = trimesh.Trimesh(vertices=vs, faces=faces)
            return trimesh.exchange.gltf.export_glb(mesh, include_normals=True)
        elif return_format == 'obj':
            return files_utils.return_obj_mesh(mesh)

    @models_utils.torch_no_grad
    def generate_random_chair(self, res=220, return_format='gltf', num_samples=1):
        """Generate random chair(s).
        
        Args:
            res: Resolution of the mesh
            return_format: Either 'gltf' or 'mesh'
            num_samples: Number of chairs to generate
            
        Returns:
            Single result or list of results in specified format
        """
        print("Starting random chair generation")
        zh, gmms = self.model.random_samples(num_samples)
        print(f"Random samples generated: zh shape={zh.shape if hasattr(zh, 'shape') else 'no shape'}")
        results = []
        
        for i in range(num_samples):
            print(f"Getting mesh for sample {i}")
            mesh = self.get_mesh(zh[i], res, None)
            print(f"Mesh result: {type(mesh)}, {mesh}")
            vs, faces = mesh
            print(f"Vertices shape: {vs.shape}, Faces shape: {faces.shape}")
            mesh = trimesh.Trimesh(vertices=vs, faces=faces)
            
            if return_format == 'gltf':
                result = trimesh.exchange.gltf.export_glb(mesh, include_normals=True)
            else:
                result = mesh
            results.append(result)
            
        # return "dummy output string"
        return results[0] if num_samples == 1 else results

    @models_utils.torch_no_grad  
    def generate_random_parts(self, chair_id: int, replace_inds: List[int], res=220, num_samples=1):
        """Generate chair(s) with random parts.
        
        Args:
            chair_id: Base chair ID
            replace_inds: Indices of parts to randomize
            res: Resolution of the mesh
            num_samples: Number of variations to generate
            
        Returns:
            Single GLTF or list of GLTFs
        """
        zh, gmms, _, _ = self.loaded_files_dict.get(chair_id)
        replace_inds_t = torch.tensor(replace_inds, dtype=torch.int64, device=self.device)
        
        # Get random embeddings
        select_random = torch.rand(1000).argsort()[:1]
        zh_new, _, gmms_new, _ = self.model.get_embeddings(select_random.to(self.device))
        zh_new = zh_new[:, replace_inds_t]
        gmms_new = [item[:, :, replace_inds_t] for item in gmms_new[0]]
        
        results = []
        for i in range(num_samples):
            # Combine embeddings
            zh_ = torch.cat((zh, zh_new[i].unsqueeze(0)), dim=1)
            gmms_ = [torch.cat((item, item_new[i].unsqueeze(0)), dim=2) 
                    for item, item_new in zip(gmms, gmms_new)]
                    
            # Generate mesh
            zh_ = self.model.merge_zh_step_a(zh_, [gmms_])
            zh_, _ = self.model.affine_transformer.forward_with_attention(zh_)
            mesh = self.get_mesh(zh_[0], res, None)
            vs, faces = mesh
            
            # Convert to GLTF
            mesh = trimesh.Trimesh(vertices=vs, faces=faces)
            result = trimesh.exchange.gltf.export_glb(mesh, include_normals=True)
            results.append(result)
            
        return results[0] if num_samples == 1 else results

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


# Initialize Flask app with better configuration
app = Flask(__name__)
CORS(app)


# Error handling
@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"An error occurred: {str(error)}", exc_info=True)
    return jsonify({
        'error': str(error),
        'status': 'error'
    }), 500


# Initialize the inference model
try:
    # Check GPU status before initializing
    has_gpu = check_gpu_status()

    # Initialize with GPU if available, otherwise use CPU
    device = CUDA(0) if has_gpu else "cpu"
    opt_ = Options(device=device, tag='chairs_sym_hard', model_name='occ_gmm').load()
    inference = Inference(opt_)

    # Load model files
    names = files_utils.collect(f"{opt_.cp_folder}/fresh_mix_origin/", '.pkl')
    paths = [''.join(item) for item in names]
    sample_names = [name[1] for name in names]
    sample_numbers = [int(s.split("_")[1]) for s in sample_names]
    inference.init_dict(paths, sample_numbers)
    num_to_path_dict = dict(zip(sample_numbers, paths))

    logger.info("Model initialization successful")
except Exception as e:
    logger.error("Failed to initialize the model", exc_info=True)
    raise

def get_numbers():
    return inference.get_indices()

def get_prediction():
    return inference.plot_chairs(chair_ids=[5047], return_result=True)

def get_mix(chairs, alphas, return_format='gltf'):
    return inference.mix_files(chairs, alphas=alphas, res=256, return_format=return_format)

def get_random_parts(chair: int, ind_to_randomize: int, num_samples: int = 1) -> Union[bytes, List[bytes]]:
    """Generate chair(s) with random parts.
    
    Args:
        chair: Base chair ID
        ind_to_randomize: Index of part to randomize (0=top, 1=mid, 2=bottom)
        num_samples: Number of variations to generate
        
    Returns:
        Single GLTF bytes or list of GLTF bytes depending on num_samples
    """
    top_i, mid_i, bottom_i = inference.loaded_files_indices_dict.get(chair)
    parts_to_randomize = [top_i, mid_i, bottom_i][ind_to_randomize]
    return inference.generate_random_parts(chair, parts_to_randomize, res=256, num_samples=num_samples)

def get_random_chair(num_samples: int = 1, return_format: str = 'gltf') -> Union[trimesh.Trimesh, bytes, List[bytes]]:
    """Generate random chair(s).
    
    Args:
        num_samples: Number of chairs to generate
        return_format: Either 'gltf' or 'mesh'
        
    Returns:
        - If return_format='mesh': Single trimesh.Trimesh
        - If return_format='gltf': Single GLTF bytes or list of GLTF bytes depending on num_samples
    """
    return inference.generate_random_chair(res=256, return_format=return_format, num_samples=num_samples)

def save_chair_png(mesh, path):
    trimesh_scene = trimesh.Scene(geometry=mesh)
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

def print_gpu_info():
    """Print detailed GPU memory usage information"""
    if torch.cuda.is_available():
        logger.info(f"\nGPU Memory Usage:")
        logger.info(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

def verify_model_on_gpu(model):
    """Verify that the model is on GPU"""
    return next(model.parameters()).device.type == 'cuda'

def verify_tensor_on_gpu(tensor):
    """Verify that a tensor is on GPU"""
    return tensor.device.type == 'cuda'

@app.route('/predict')
def predict():
    result = get_prediction()
    return jsonify({'path': result})

@app.route('/mix', methods=['POST'])
def mix():
    """Mix multiple chair designs together based on provided blend weights.
    
    Expects POST data with:
    - name: Name of the design
    - chair_[1-4]: IDs of the 4 base chairs
    - chair_[1-4]_[top/mid/bottom]: Blend weights for each chair section
    
    Returns the mixed chair as a GLTF file.
    """
    global restart_counter
    global inference
    
    # Parse request data
    data = request.get_json()  # Remove the json.loads() call
    print(data)
    chairs = [int(data.get(f'chair_{i}', '')) for i in range(1,5)]
    alphas = [
        inference.get_alpha(
            chairs[i-1],
            int(data.get(f'chair_{i}_bottom', '')),
            int(data.get(f'chair_{i}_mid', '')), 
            int(data.get(f'chair_{i}_top', ''))
        ) for i in range(1,5)
    ]
        
    return get_mix(chairs, alphas)

@app.route('/random_part', methods=['POST'])
def random_part():
    """Randomize a specific part of a chair design.
    
    Expects POST data with:
    - name: Name of the design 
    - chair: ID of the base chair
    - random_index: Index of the part to randomize
    
    Returns the modified chair as a GLTF file.
    """
    data = request.get_json()  # This already gives you parsed JSON
    chair = int(data.get('chair', ''))
    ind_to_randomize = int(data.get('random_index', ''))
    return get_random_parts(chair, ind_to_randomize)

@app.route('/random_chair', methods=['POST'])
def random_chair():
    result = get_random_chair(num_samples = 1)
    return Response(
        result,
        mimetype='model/gltf-binary',
        headers={
            'Content-Type': 'model/gltf-binary',
            'Content-Disposition': 'attachment; filename=random_chair.glb'
        }
    )
    # return get_random_chair(num_samples = 1)


@app.route('/archive', methods=['POST'])
def archive():
    """Archive a chair design by saving its mesh, thumbnail image, and metadata.
    
    Expects POST data with:
    - name: Name of the design
    - creator: Creator's name 
    - chair_[1-4]: IDs of the 4 base chairs
    - chair_[1-4]_[top/mid/bottom]: Blend weights for each chair section
    """
    # Parse request data
    data = request.get_json().get('body', '')
    data = json.loads(data)
    name = data.get('name', '')
    creator = data.get('creator', '')
    
    # Get chair IDs and calculate blend weights
    chairs = [int(data.get(f'chair_{i}', '')) for i in range(1,5)]
    alphas = [
        inference.get_alpha(
            chairs[i-1],
            int(data.get(f'chair_{i}_bottom', '')),
            int(data.get(f'chair_{i}_mid', '')), 
            int(data.get(f'chair_{i}_top', ''))
        ) for i in range(1,5)
    ]

    # Generate and save the mesh
    mesh = inference.mix_files(chairs, alphas)
    gltf_to_save = trimesh.exchange.gltf.export_glb(mesh, include_normals=True)
    
    # Save metadata to archive list
    file_name = f"{inference.opt.archive_json_folder}/ArchiveList.json"
    with open(file_name) as fp:
        archive_list = json.load(fp)
        archive_list.append({
            "itemName": name,
            "year": "2024", 
            "creator": f"{creator} X chAIr"
        })
    
    with open(file_name, 'w') as json_file:
        json.dump(archive_list, json_file, indent=4, separators=(',', ': '))

    # Save mesh file and thumbnail image
    with open(f"{inference.opt.archive_model_folder}/{name}.gltf", 'wb') as f:
        f.write(gltf_to_save)
    return save_chair_png(mesh, path=f"{inference.opt.archive_image_folder}/chair_{name}.png")

@app.route('/archive_random', methods=['POST']) 
def archive_random():
    """Archive a randomly generated chair design.
    
    Similar to /archive endpoint but generates a random mesh instead of blending existing chairs.
    """
    # Parse request data
    data = request.get_json().get('body', '')
    data = json.loads(data)
    name = data.get('name', '')
    creator = data.get('creator', '')

    # Generate random chair mesh
    mesh = get_random_chair(return_format='mesh')
    gltf_to_save = trimesh.exchange.gltf.export_glb(mesh, include_normals=True)

    # Save metadata to archive list
    file_name = f"{inference.opt.archive_json_folder}/ArchiveList.json"
    with open(file_name) as fp:
        archive_list = json.load(fp)
        archive_list.append({
            "itemName": name,
            "year": "2024",
            "creator": f"{creator} X chAIr"
        })

    with open(file_name, 'w') as json_file:
        json.dump(archive_list, json_file, indent=4, separators=(',', ': '))

    # Save mesh file and thumbnail image
    with open(f"{inference.opt.archive_model_folder}/{name}.gltf", 'wb') as f:
        f.write(gltf_to_save)
    return save_chair_png(mesh, path=f"{inference.opt.archive_image_folder}/chair_{name}.png")

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.get('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })

@app.route('/gpu_status')
def gpu_status():
    """Endpoint to check GPU status and memory usage"""
    if torch.cuda.is_available():
        return jsonify({
            'gpu_available': True,
            'gpu_name': torch.cuda.get_device_name(0),
            'memory_allocated': f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB",
            'memory_reserved': f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB",
            'max_memory_reserved': f"{torch.cuda.max_memory_reserved(0) / 1024**2:.2f} MB",
            'model_on_gpu': verify_model_on_gpu(inference.model)
        })
    return jsonify({
        'gpu_available': False,
        'error': 'No GPU available'
    })

if __name__ == '__main__':
    # Print system status
    logger.info("Starting server...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Current working directory: {os.getcwd()}")

    # Start the Flask server
    app.run(threaded=False)