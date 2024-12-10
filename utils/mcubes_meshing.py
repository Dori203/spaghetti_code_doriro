import skimage.measure
import time
from custom_types import *
from utils.train_utils import Logger
import constants


def mcubes_skimage(pytorch_3d_occ_tensor: T, voxel_grid_origin: List[float], voxel_size: float) -> T_Mesh:
    print(f"Input tensor shape: {pytorch_3d_occ_tensor.shape}")
    print(
        f"Input tensor stats - min: {pytorch_3d_occ_tensor.min()}, max: {pytorch_3d_occ_tensor.max()}, mean: {pytorch_3d_occ_tensor.mean()}")
    print(f"Voxel grid origin: {voxel_grid_origin}, voxel size: {voxel_size}")

    numpy_3d_occ_tensor = pytorch_3d_occ_tensor.numpy()
    print(f"Numpy tensor shape: {numpy_3d_occ_tensor.shape}")
    print(
        f"Numpy tensor stats - min: {numpy_3d_occ_tensor.min()}, max: {numpy_3d_occ_tensor.max()}, mean: {numpy_3d_occ_tensor.mean()}")

    try:
        marching_cubes = skimage.measure.marching_cubes if 'marching_cubes' in dir(
            skimage.measure) else skimage.measure.marching_cubes_lewiner
        print(f"Using marching cubes function: {marching_cubes.__name__}")
        # Calculate level based on data range
        data_min = numpy_3d_occ_tensor.min()
        data_max = numpy_3d_occ_tensor.max()
        level = (data_max + data_min) / 2  # or use a different strategy to set the level
        print(f"Using level value: {level} for range [{data_min}, {data_max}]")

        verts, faces, normals, values = marching_cubes(numpy_3d_occ_tensor, level=level, spacing=[voxel_size] * 3)
        print(f"Marching cubes successful - vertices: {verts.shape}, faces: {faces.shape}")

    except Exception as e:
        print("mc failed")
        print(f"Marching cubes error: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

    # transform from voxel coordinates to camera coordinates
    print("Transforming coordinates...")
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    print(f"Final mesh points shape: {mesh_points.shape}")
    return torch.from_numpy(mesh_points.copy()).float(), torch.from_numpy(faces.copy()).long()


class MarchingCubesMeshing:

    # def fill_samples(self, decoder, samples, device: Optional[D] = None) -> T:
    #     num_samples = samples.shape[0]
    #     num_iters = num_samples // self.max_batch + int_b(num_samples % self.max_batch != 0)
    #     if self.verbose:
    #         logger = Logger()
    #         logger.start(num_iters, tag='meshing')
    #     for i in range(num_iters):
    #         sample_subset = samples[i * self.max_batch: min((i + 1) * self.max_batch, num_samples), 0:3]
    #         if device is not None:
    #             sample_subset = sample_subset.to(device)
    #         samples[i * self.max_batch: min((i + 1) * self.max_batch, num_samples), 3] = (
    #             decoder(sample_subset * self.scale).squeeze().detach()
    #         )
    #         if self.verbose:
    #             logger.reset_iter()
    #     if self.verbose:
    #         logger.stop()
    #     return samples

    def fill_samples(self, decoder, samples, device: Optional[D] = None) -> T:
        print(f"\nFill samples - input shape: {samples.shape}")
        print(f"Scale value: {self.scale}")
        num_samples = samples.shape[1]
        print(f"num samples: {num_samples}")

        num_iters = num_samples // self.max_batch + int(num_samples % self.max_batch != 0)
        sample_coords = samples[:3]
        if self.verbose:
            logger = Logger()
            logger.start(num_iters, tag='meshing')
        for i in range(num_iters):
            sample_subset = sample_coords[:, i * self.max_batch: min((i + 1) * self.max_batch, num_samples)]
            print(f"\nBatch {i} - subset shape before scaling: {sample_subset.shape}")
            print(f"Subset stats before scaling - min: {sample_subset.min()}, max: {sample_subset.max()}")
            if device is not None:
                sample_subset = sample_subset.to(device)
                print(f"Subset stats after scaling to device - min: {sample_subset.min()}, max: {sample_subset.max()}")
            sample_subset = sample_subset.T
            samples[3, i * self.max_batch: min((i + 1) * self.max_batch, num_samples)] = (
                decoder(sample_subset * self.scale).squeeze().detach()
            )
            if self.verbose:
                logger.reset_iter()
        if self.verbose:
            logger.stop()
        return samples

    def fill_recursive(self, decoder, samples: T, stride: int, base_res: int, depth: int) -> T:
        print(f"\nFill recursive - base_res: {base_res}, depth: {depth}")
        if base_res <= self.min_res:
            print("Hit min resolution")
            samples_ = self.fill_samples(decoder, samples)
            return samples_
        kernel_size = 7 + 4 * depth
        padding = tuple([kernel_size // 2] * 6)
        # samples_ = samples.transpose(1, 0).view(1, 4, base_res, base_res, base_res)
        # samples_ = nnf.avg_pool3d(samples_, stride, stride)
        # samples_ = samples_.view(4, -1).transpose(1, 0)
        # res = base_res // stride
        # samples_lower = self.fill_recursive(decoder, samples_, stride, res)
        # mask = samples_lower[:, -1].lt(.3)
        # mask = mask.view(1, 1, res, res, res).float()
        # mask = nnf.pad(mask, padding, mode='replicate')
        # mask = nnf.max_pool3d(mask, kernel_size, 1)
        # mask = nnf.interpolate(mask, scale_factor=stride)
        # mask = mask.flatten().bool()
        # samples[mask] = self.fill_samples(decoder, samples[mask])

        samples_ = samples.view(1, 4, base_res, base_res, base_res)
        samples_ = nnf.avg_pool3d(samples_, stride, stride)
        samples_ = samples_.view(4, -1)
        res = base_res // stride
        print(f"After pooling - res: {res}")

        samples_lower = self.fill_recursive(decoder, samples_, stride, res, depth - 1)
        print(
            f"samples_lower stats - min: {samples_lower[-1, :].min()}, max: {samples_lower[-1, :].max()}, mean: {samples_lower[-1, :].mean()}")

        print(f"After recursive call - samples_lower shape: {samples_lower.shape}")

        mask = samples_lower[-1, :].lt(.3)

        print(f"Mask sum: {mask.sum()} out of {mask.numel()} (ratio: {mask.sum() / mask.numel():.3f})")

        mask = mask.view(1, 1, res, res, res).float()
        mask = nnf.pad(mask, padding, mode='replicate')
        mask = nnf.max_pool3d(mask, kernel_size, 1)
        mask = nnf.interpolate(mask, scale_factor=stride)
        mask = mask.flatten().bool()
        samples[:, mask] = self.fill_samples(decoder, samples[:, mask])
        return samples

    def tune_resolution(self, res: int):
        counter = 1
        while res > self.min_res:
            res = res // 2
            counter *= 2
        return res * counter

    @staticmethod
    def get_res_samples(res):
        voxel_origin = torch.tensor([-1., -1., -1.])
        voxel_size = 2.0 / (res - 1)
        overall_index = torch.arange(0, res ** 3, 1, dtype=torch.int64)
        samples = torch.ones(4, res ** 3).detach()
        samples.requires_grad = False
        # transform first 3 columns
        # to be the x, y, z index
        div_1 = torch.div(overall_index, res, rounding_mode='floor')
        samples[2, :] = (overall_index % res).float()
        samples[1, :] = (div_1 % res).float()
        samples[0, :] = (torch.div(div_1, res, rounding_mode='floor') % res).float()
        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:3] = samples[:3] * voxel_size + voxel_origin[:, None]
        # samples[0, :] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        # samples[1, :] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        # samples[2, :] = (samples[:, 2] * voxel_size) + voxel_origin[0]
        return samples

    def register_resolution(self, res: int):
        res = self.tune_resolution(res)
        if res not in self.sample_cache:
            samples = self.get_res_samples(res)
            samples = samples.to(self.device)
            self.sample_cache[res] = samples
        else:
            samples = self.sample_cache[res]
            samples[3, :] = 1
        return samples, res

    def get_grid(self, decoder, res):
        print(f"Initial res: {res}")
        stride = 2
        samples, res = self.register_resolution(res)
        print(f"After register_resolution - samples shape: {samples.shape}, new res: {res}")

        depth = int(np.ceil(np.log2(res) - np.log2(self.min_res)))
        print(f"Calculated depth: {depth}, min_res: {self.min_res}")

        samples = self.fill_recursive(decoder, samples, stride, res, depth)
        print(f"After fill_recursive - samples shape: {samples.shape}")

        occ_values = samples[3]
        occ_values = occ_values.reshape(res, res, res)
        print(f"Final occ_values shape: {occ_values.shape}")

        return occ_values

    def occ_meshing(self, decoder, res: int = 256, get_time: bool = False, verbose=False):
        try:
            start = time.time()
            voxel_origin = [-1., -1., -1.]
            voxel_size = 2.0 / (res - 1)

            # Get grid values
            print("Getting grid values...")
            occ_values = self.get_grid(decoder, res)
            print(f"Grid shape: {occ_values.shape}, min: {occ_values.min()}, max: {occ_values.max()}")

            if verbose:
                end = time.time()
                print("sampling took: %f" % (end - start))
                if get_time:
                    return end - start

            # Move to CPU and check for valid values
            occ_data = occ_values.data.cpu()
            print(f"CPU data shape: {occ_data.shape}, any NaN: {torch.isnan(occ_data).any()}")

            # Run marching cubes
            print("Starting marching cubes...")
            try:
                mesh_a = mcubes_skimage(occ_data, voxel_origin, voxel_size)
                print(f"Marching cubes successful, mesh vertices: {mesh_a[0].shape}, faces: {mesh_a[1].shape}")
            except Exception as mc_error:
                print(f"Marching cubes failed with error: {str(mc_error)}")
                print(f"Data stats - min: {occ_data.min()}, max: {occ_data.max()}, mean: {occ_data.mean()}")
                raise

            if verbose:
                end_b = time.time()
                print("mcube took: %f" % (end_b - end))
                print("meshing took: %f" % (end_b - start))

            return mesh_a

        except Exception as e:
            print(f"Error in occ_meshing: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None

    def __init__(self, device: D, max_batch: int = 64 ** 3, min_res: int = 128, scale: float = 1,
                 max_num_faces: int = 0, verbose: bool = False):
        self.device = device
        self.max_batch = max_batch
        self.max_batch = 32 ** 3 # if constants.IS_WINDOWS else max_batch
        self.min_res = min_res
        self.scale = scale
        self.verbose = verbose
        self.sample_cache = {}
        self.max_num_faces = max_num_faces


def create_mesh_old(decoder, res=256, max_batch=64 ** 3, scale=1, device=CPU, verbose=False, get_time: bool = False):
    meshing = MarchingCubesMeshing(device, max_batch=max_batch, scale=scale, verbose=verbose)
    start = time.time()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (res - 1)

    overall_index = torch.arange(0, res ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(res ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % res
    samples[:, 1] = (overall_index.long() // res) % res
    samples[:, 0] = ((overall_index.long() // res) // res) % res

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    samples = meshing.fill_samples(decoder, samples, device=device)
    sdf_values = samples[:, 3]
    # return sdf_values, samples[:, :3]
    sdf_values = sdf_values.reshape(res, res, res)

    end = time.time()
    print("sampling took: %f" % (end - start))
    if get_time:
        return end - start
    return mcubes_skimage(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
    )
