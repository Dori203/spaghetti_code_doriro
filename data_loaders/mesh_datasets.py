import models.models_utils
from custom_types import *
from utils import files_utils, mesh_utils
import constants
from utils import train_utils
from threading import Thread
# import make_data
import abc
import h5py
import options
import zipfile
import os


class OccDataset(Dataset, abc.ABC):

    def __len__(self):
        return len(self.paths)

    @abc.abstractmethod
    def get_samples_(self, item: int, total: float) -> TS:
        raise NotImplemented

    def get_samples(self, item: int, total: float = 5e5) -> TS:
        if self.data[item] is None:
            self.data[item] = self.get_samples_(item, total)
        return self.data[item]

    @staticmethod
    def shuffle_(points: T, labels: Optional[T] = None) -> Union[T, TS]:
        order = torch.rand(points.shape[0], device=points.device).argsort()
        if labels is None:
            return points[order]
        return points[order], labels[order]

    @abc.abstractmethod
    def shuffle(self, item: int, *args):
        raise NotImplemented

    def sampler(self, item, points, labels):
        points_ = points[:, self.counter[item] * self.num_samples: (1 + self.counter[item]) * self.num_samples]
        labels_ = labels[:, self.counter[item] * self.num_samples: (1 + self.counter[item]) * self.num_samples]
        self.counter[item] += 1
        if (self.counter[item] + 1) * self.num_samples > points.shape[1]:
            self.counter[item] = 0
            self.shuffle(item)
        return points_, labels_

    def get_large_batch(self, item: int, num_samples: int):
        points, labels = self.get_samples(item)
        select = torch.rand(points.shape[1]).argsort()[:num_samples]
        points, labels = points[:, select], labels[:, select]
        return points, labels

    def __getitem__(self, item: int):
        points, labels = self.get_samples(item)
        points, labels = self.sampler(item, points, labels)
        for axis in self.symmetric_axes:
            points_ = points.clone()
            points_[:, :, axis] = -points_[:, :, axis]
            points = torch.cat((points, points_), dim=1)
            labels = torch.cat((labels, labels), dim=1)
        return points, labels, item

    @staticmethod
    @abc.abstractmethod
    def collect(ds_name: str) -> List[List[str]]:
        raise NotImplemented

    def filter_paths(self, paths: List[List[str]]) -> List[List[str]]:
        if self.split_path is not None:
            names = files_utils.load_json(self.split_path)["ShapeNetV2"]
            names = list(names.items())[0][1]
            paths = list(filter(lambda x: x[1] in names, paths))
        return paths

    def __init__(self, ds_name: str, num_samples: int, symmetric: Tuple[bool, bool, bool], split_path: Optional[str] = None):
        self.split_path = split_path
        self.num_samples = num_samples
        paths = self.collect(ds_name)
        self.paths = self.filter_paths(paths)
        self.data: List[TSN] = [None] * len(self)
        self.counter = [0] * len(self)
        self.symmetric_axes = [i for i in range(len(symmetric)) if symmetric[i]]


class MeshDataset(OccDataset):

    def load_mesh(self, item: int):
        mesh = files_utils.load_mesh(''.join(self.paths[item]))
        mesh = mesh_utils.triangulate_mesh(mesh)[0]
        mesh = mesh_utils.to_unit_sphere(mesh)
        return mesh

    def get_samples_(self, item: int, total: float) -> TS:
        mesh = self.load_mesh(item)
        on_surface_points = mesh_utils.sample_on_mesh(mesh, int(total * self.split[0]))[0]
        if self.split[1] > 0:
            near_points = on_surface_points + torch.randn_like(on_surface_points) * .01
            random_points = torch.rand(int(total * self.split[1]), 3) * 2 - 1
            all_points = torch.cat((on_surface_points, near_points, random_points), dim=0)
            labels_near = mesh_utils.get_inside_outside(near_points, mesh)
            labels_random = mesh_utils.get_inside_outside(random_points, mesh)
            labels = torch.cat((torch.zeros(on_surface_points.shape[0]), labels_near, labels_random), dim=0)
        else:
            all_points = on_surface_points
            labels = self.labels
        shuffle = torch.argsort(torch.rand(int(total)))
        return all_points[shuffle], labels[shuffle]

    @staticmethod
    def collect(ds_name: str) -> List[List[str]]:
        return files_utils.collect(f'{constants.RAW_ROOT}{ds_name}/', '.obj', '.off')

    def __init__(self, ds_name: str, num_samples: int, flow: int):
        super(MeshDataset, self).__init__(ds_name, num_samples)
        if flow == 1:
            self.split = (.4, .4, .2)
        else:
            self.split = (1., .0, .0)
            self.labels = torch.zeros(int(5e5))


class SingleMeshDataset(OccDataset):

    def __getitem__(self, item: int):
        points, labels, _ = super(SingleMeshDataset, self).__getitem__(0)
        return points, labels, 0

    def __len__(self):
        return self.single_labels.shape[1] // self.num_samples

    def get_samples_(self, _: int, __: float) -> TS:
        return self.single_points, self.single_labels

    def shuffle(self, item: int, *args):
        all_points, labels = self.single_points, self.single_labels
        shuffled = [self.shuffle_(all_points[i], labels[i]) for i in range(labels.shape[0])]
        all_points = torch.stack([item[0] for item in shuffled], dim=0)
        labels = torch.stack([item[1] for item in shuffled], dim=0)
        self.single_points, self.single_labels = all_points, labels

    @staticmethod
    def init_samples(mesh_name: str, symmetric, device: D) -> TS:
        mesh_path = f'{constants.DATA_ROOT}singles/{mesh_name}'
        if not files_utils.is_file(mesh_path + '.npz'):
            sampler = make_data.MeshSampler(mesh_path, CUDA(0), (make_data.ScaleType.Sphere, None, 1.), inside_outside=True,
                                  symmetric=symmetric, num_samples=5e6)
            points, labels = sampler.points, sampler.labels
            data = {'points': points.cpu().numpy(), 'labels': labels.cpu().numpy()}
            if not sampler.error:
                files_utils.save_np(data, mesh_path)
            else:
                print(f'error: {mesh_path}')
        else:
            data: Dict[str, ARRAY] = np.load(mesh_path + '.npz')
        all_points, labels = torch.from_numpy(data['points']), torch.from_numpy(data['labels'])
        return all_points.to(device), labels.to(device)

    @staticmethod
    def collect(mesh_name: str) -> List[List[str]]:
        return []

    def __init__(self,  mesh_name: str, num_samples: int, symmetric: Tuple[bool, bool, bool], device: D):
        self.device = device
        self.single_points, self.single_labels = self.init_samples(mesh_name, symmetric, device)
        super(SingleMeshDataset, self).__init__(mesh_name, num_samples, symmetric)


class CacheDataset(OccDataset):

    # def sampler(self, item, points, labels):
    #     if self.sampler_by is None:
    #         return super(CacheDS, self).sampler(item, points, labels)
    #     surface_inds = np.random.choice(self.sampler_data[item][0], self.num_samples // 2, replace=False)
    #     random_inds = np.random.choice(self.sampler_data[item][1], self.num_samples - self.num_samples // 2, replace=False)
    #     points = torch.cat((points[surface_inds], points[random_inds]), dim=0)
    #     labels = torch.cat((labels[surface_inds], labels[random_inds]))
    #     return points, labels

    def get_name_mapper(self) -> Dict[str, int]:
        if self.name_mapper is None:
            self.name_mapper = {self.get_name(item): item for item in range(len(self))}
        return self.name_mapper

    def get_item_by_name(self, name: str) -> int:
        return self.get_name_mapper()[name]

    def get_name(self, item: int):
        return self.paths[item][1]

    def shuffle(self, item: int, *args):
        return
    # def shuffle(self, item: int, *args):
    #     all_points, labels = self.data[item]
    #     shuffled = [super(CacheDataset, self).shuffle_(all_points[i], labels[i]) for i in range(labels.shape[0])]
    #     all_points = torch.stack([item[0] for item in shuffled], dim=0)
    #     labels = torch.stack([item[1] for item in shuffled], dim=0)
    #     self.data[item] = all_points, labels

    def get_samples_(self, item: int, _) -> TS:
        path = f'{self.root}{self.get_name(item)}'
        all_points = np.load(f"{path}_pts.npy")
        labels = np.load(f"{path}_lbl.npy")
        # data: Dict[str, ARRAY] = np.load(''.join(self.paths[item]))
        # all_points, labels = torch.from_numpy(data['points']), torch.from_numpy(data['labels'])
            # if self.sampler_by is None:
            #     shuffle = torch.argsort(torch.rand(int(labels.shape[0])))
            #     all_points, labels = all_points[shuffle], labels[shuffle]
            # on_surface_ind = labels.abs().lt(0.02)
            # near_surface_ind = torch.where(~on_surface_ind)[0].numpy()
            # on_surface_ind = torch.where(on_surface_ind)[0].numpy()
            # self.sampler_data[item] = (on_surface_ind, near_surface_ind)
            # self.data[item] = all_points, labels
        return torch.from_numpy(all_points), torch.from_numpy(labels)


    @staticmethod
    def collect(ds_name: str) -> List[List[str]]:
        files = files_utils.collect(f'{constants.CACHE_ROOT}inside_outside/{ds_name}/', '.npy')
        files = [file for file in files if '_pts' in file[1]]
        files = [[file[0], file[1][:-4], file[2]] for file in files]
        return files

    def __init__(self, ds_name: str, num_samples: int, symmetric: Tuple[bool, bool, bool],
                 split_path: Optional[str] = None):
        self.root = f'{constants.CACHE_ROOT}inside_outside/{ds_name}/'
        super(CacheDataset, self).__init__(ds_name, num_samples, symmetric, split_path)
        self.sampler_data: List[TSN] = [None] * len(self)
        self.name_mapper: Optional[Dict[str, int]] = None


class CacheImNet(CacheDataset):

    def __getitem__(self, item):
        points, labels, _ = super(CacheImNet, self).__getitem__(item)
        vox = np.load(f"{self.im_net_root}{self.get_name(item)}.npy")
        vox = 1 - 2 * vox
        # try:
        #
        # except ValueError:
        #     files_utils.delete_single(f"{self.im_net_root}{self.get_name(item)}.npy")
        #     raise BaseException
        vox = torch.from_numpy(vox).view(1, 64, 64, 64).float()
        return points, labels, vox, item

    def filter_paths(self, paths: List[List[str]]) -> List[List[str]]:
        names_im_net = set(map(lambda x: x[1], files_utils.collect(self.im_net_root, '.npy')))
        paths = filter(lambda x: x[1] in names_im_net, paths)
        return list(paths)

    def __init__(self, cls: str, ds_name: str, num_samples: int, symmetric: Tuple[bool, bool, bool]):
        self.im_net_root = f'{constants.CACHE_ROOT}im_net/{cls}/'
        super(CacheImNet, self).__init__(ds_name, num_samples, symmetric)


class CacheInOutDataset(CacheDataset):

    def shuffle(self, item: int, *args):
        inside_points = self.data[item][0][3]
        inside_points = self.shuffle_(inside_points).unsqueeze_(0)
        super(CacheInOutDataset, self).shuffle(item, *args)
        all_points, labels = self.data[item]
        all_points = torch.cat((all_points, inside_points), dim=0)
        self.data[item] = all_points, labels

    @staticmethod
    def collect(ds_name: str, split_path: Optional[str] = None) -> List[List[str]]:
        return files_utils.collect(f'{constants.CACHE_ROOT}inside_outside/{ds_name}/', '.npz')


class CacheH5Dataset(CacheDataset):

    def get_name(self, item: int):
        return self.names[item]

    def get_samples(self, item: int, total: float = 5e5) -> TS:
        dataset = self.dataset[0]
        for i in range(len(self.lengths)):
            if item < self.lengths[i]:
                dataset = self.dataset[i]
                break
            item -= self.lengths[i]
        return torch.from_numpy(dataset['points'][item]).float(), torch.from_numpy(dataset['labels'] [item]).float()

    def shuffle(self, item: int, *args):
        return

    def __len__(self):
        return sum(self.lengths)

    @staticmethod
    def collect(ds_name: str) -> List[List[str]]:
        return files_utils.collect(f'{constants.CACHE_ROOT}inside_outside/{ds_name}/', '.hdf5')

    def __init__(self, ds_name: str, num_samples: int, symmetric: Tuple[bool, bool, bool]):
        paths = files_utils.collect(f'{constants.CACHE_ROOT}inside_outside/{ds_name}/', '.hdf5')
        self.dataset = [h5py.File(''.join(path), "r") for path in paths]
        self.lengths = [dataset['points'].shape[0] for dataset in self.dataset]
        super(CacheH5Dataset, self).__init__(ds_name, num_samples, symmetric)
        self.names = files_utils.load_pickle(f'{constants.CACHE_ROOT}inside_outside/{ds_name}/all_data_names')


class SimpleDataLoader:

    @staticmethod
    def tensor_reducer(items: TS) -> T:
        return torch.stack(items, dim=0)

    @staticmethod
    def number_reducer(dtype) -> Callable[[List[float]], T]:
        def reducer_(items: List[float]) -> T:
            return torch.tensor(items, dtype=dtype)
        return reducer_

    @staticmethod
    def default_reducer(items: List[Any]):
        return items

    def __iter__(self):
        self.counter = 0
        self.order = torch.rand(len(self.ds)).argsort()
        return self

    def __len__(self):
        return self.length

    def collate_fn(self, raw_data):
        batch = [self.reducers[i](items) for i, items in enumerate(zip(*raw_data))]
        return batch

    def init_reducers(self, sample):
        reducers = []
        for item in sample:
            if type(item) is T:
                reducers.append(self.tensor_reducer)
            elif type(item) is int:
                reducers.append(self.number_reducer(torch.int64))
            elif type(item) is float:
                reducers.append(self.number_reducer(torch.float32))
            else:
                reducers.append(self.default_reducer)
        return reducers

    def __next__(self):
        if self.counter < len(self):
            start = self.counter * self.batch_size
            indices = self.order[start:  min(start + self.batch_size, len(self.ds))].tolist()
            raw_data = [self.ds[ind] for ind in indices]
            self.counter += 1
            return self.collate_fn(raw_data)
        else:
            raise StopIteration

    def __init__(self, ds: Dataset, batch_size: int = 1):
        self.ds = ds
        self.batch_size = batch_size
        self.counter = 0
        self.length = len(ds) // self.batch_size + int(len(self.ds) % self.batch_size != 0)
        self.order: Optional[T] = None
        self.reducers = self.init_reducers(self.ds[0])


def save_np_mesh_thread(name: str, items: T, logger: train_utils.Logger):
    ds = MeshDataset(name, 512, 1)
    root = f'{constants.RAW_ROOT}shapenet_numpy/{name}/'
    items = items.tolist()
    for i in items:
        item_id = ds.paths[i][0].split('/')[-3]
        save_path = f'{root}{item_id}'
        if not files_utils.is_file(save_path + '.npy'):
            mesh = ds.load_mesh(i)
            np_mesh = (mesh[0][mesh[1]]).numpy()
            files_utils.save_np(np_mesh, save_path)
        logger.reset_iter()


def save_np_mesh(name):
    ds = MeshDataset(name, 512, 1)
    logger = train_utils.Logger().start(len(ds))
    num_threads = 4
    split_size = len(ds) // num_threads
    threads = []
    for i in range(num_threads):
        if i == num_threads - 1:
            split_size_ = len(ds) - (num_threads - 1) * split_size
        else:
            split_size_ = split_size
        items = torch.arange(split_size_) + i * split_size
        threads.append(Thread(target=save_np_mesh_thread, args=(name, items, logger)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    logger.stop()



def export_points():
    opt = options.OptionsSingle()
    ds = SingleMeshDataset(opt.dataset_name, opt.num_samples,
                            opt.symmetric, opt.device)
    points = ds.single_points
    colors = [(1., 0, 0), (0, 1., 0), (0, 0, 1.), (1, .5, 0)]
    for i in range(4):
        select = torch.rand(points.shape[1]).argsort()[:10000]
        colors_ = torch.tensor(colors[i]).unsqueeze(0).expand(10000, 3)
        pts = points[i, select.numpy()]
        files_utils.export_mesh(pts, f"{constants.DATA_ROOT}/tmp/{i}", colors=colors_)


def merge_dataset(ds_name, split_name: Optional[str] = None):

    def save_tmp():
        nonlocal points
        out_path = f"{constants.DATA_ROOT}tmp/"
        for j in range(points.shape[0]):
            select = torch.rand(points.shape[1]).argsort()[:10000]
            pts = points[j, select.numpy()]
            files_utils.export_mesh(pts, f'{out_path}{i:03d}_{j}')

    def save_data(data_file):
        nonlocal all_points, all_labels
        files_utils.init_folders(data_file)
        with h5py.File(data_file, "w") as f:
            all_points = np.stack(all_points, axis=0)
            all_labels = np.stack(all_labels, axis=0)
            f["points"] = all_points
            f["labels"] = all_labels
        files_utils.save_pickle(names, f'{export_root}/all_data_names')

    if split_name is not None:
        suffixes = ("train", "test")
    else:
        suffixes = (None,)
    for suffix in suffixes:
        export_root = f'{constants.CACHE_ROOT}inside_outside/{ds_name}_{suffix}'\
            if suffix is not None else f'{constants.CACHE_ROOT}inside_outside/{ds_name}'

        split_path = f"{constants.DATA_ROOT}splits/{split_name}_{suffix}.json" if suffix is not None else None
        ds = CacheInOutDataset(ds_name, 3000, (False, False, False), split_path)
        all_points, all_labels = [], []
        logger = train_utils.Logger().start(len(ds), tag=f'{ds_name}_{suffix}')
        counter = 0
        names = []
        for i in range(len(ds)):
            # try:
            data: Dict[str, ARRAY] = np.load(''.join(ds.paths[i]))
            names.append(ds.paths[i][1])
            # except zipfile.BadZipFile:
                # files_utils.delete_single(''.join(ds.paths[i]))
                # print(ds.paths[i][1])
            # except ValueError:
                # files_utils.delete_single(''.join(ds.paths[i]))
                # print(ds.paths[i][1])
            points, labels = data['points'], data['labels']
            all_points.append(points)
            all_labels.append(labels)
            logger.reset_iter()
            save_tmp()
            if (i + 1) % 1000 == 0:
                save_data(f'{export_root}/all_data_{counter:d}.hdf5')
                counter += 1
                all_points, all_labels = [], []
        save_data(f'{export_root}/all_data_{counter:d}.hdf5')
        logger.stop()
        print("done")



def get_split_names(split_name: str, suffix: str) -> List[str]:
    split_path = f"{constants.DATA_ROOT}splits/{split_name}_{suffix}.json"
    names = files_utils.load_json(split_path)["ShapeNetV2"]
    names = list(names.items())[0][1]
    return names


def split_npy(ds_name, split_name: str):
    root = f'{constants.CACHE_ROOT}inside_outside/{ds_name}/'
    for suffix in ("test", "train"):
        export_root = f'{constants.CACHE_ROOT}inside_outside/{ds_name}_{suffix}/'
        files_utils.init_folders(export_root)
        names = get_split_names(split_name, suffix)
        for name in names:
            if files_utils.is_file(f'{root}{name}_pts.npy'):
                files_utils.move_file(f'{root}{name}_pts.npy', f'{export_root}{name}_pts.npy')
                files_utils.move_file(f'{root}{name}_lbl.npy', f'{export_root}{name}_lbl.npy')


def create_split_file(ds_name, ratio_test=0.25):

    root = f'{constants.CACHE_ROOT}inside_outside/{ds_name}/'
    names = files_utils.collect(root, '.npy')
    names = [name[1][:-4] for name in names if 'pts' in name[1]]
    split = torch.rand(len(names)).argsort()
    split = split[:6000]
    names_test = [names[split[i]] for i in range(0, int(len(split) * ratio_test))]
    names_train = [names[split[i]] for i in range(int(len(split) * ratio_test), len(split))]
    for suffix, item in zip(("test", "train"), (names_test, names_train)):
        split_path = f"{constants.DATA_ROOT}splits/shapenet_tables_{suffix}.json"
        data = {"ShapeNetV2": {"1234": item}}
        files_utils.save_json(data, split_path)
    return


def create_seg_pcd(cls: str, split_name, suffix: str = 'test'):
    path_metadata = files_utils.collect(f'{constants.PARTNET_ROOT}ins_seg_h5/{cls}/', '.json')
    root_seg = f'{constants.PARTNET_ROOT}sem_seg_h5/{cls}-1/'
    root_shapes = f"{constants.Shapenet_WT}/{cls.lower()}s/"
    names = set(get_split_names(split_name, suffix))
    logger = train_utils.Logger().start(len(names))
    counter = 0
    cls_number = get_shapenet_num_class(cls.lower())
    for metadata_path in path_metadata:
        metadata = files_utils.load_json(''.join(metadata_path))
        dataset = h5py.File(f'{root_seg}{metadata_path[1]}.h5', "r")
        labels = V(dataset['label_seg'])
        ptc = V(dataset["data"])
        for i in range(ptc.shape[0]):
            mesh_name = metadata[i]["model_id"]
            if mesh_name in names and files_utils.is_file(f'{root_shapes}{mesh_name}.obj'):
                if not files_utils.is_file(f"{constants.CACHE_ROOT}/seg/{cls.lower()}s/gt/{mesh_name}.npy"):
                    pc = torch.from_numpy(ptc[i]).float()
                    mesh_wt = files_utils.load_mesh(f'{root_shapes}{mesh_name}')
                    mesh_or = files_utils.load_mesh(f'{constants.Shapenet}{cls_number}/{mesh_name}/models/model_normalized.obj')
                    mesh_wt = mesh_utils.scale_by_ref(mesh_wt, mesh_or)
                    # tmp = pc_or[:, 2].clone()
                    # pc_or[:, 2] = pc_or[:, 0]
                    # pc_or[:, 0] = tmp
                    pc[:, 2] = - pc[:, 2]
                    pc[:, 0] = - pc[:, 0]

                    pc = mesh_closest_point.vs_cp(pc, mesh_wt)[0]
                    # pc = mesh_utils.to_unit_sphere((pc, None), scale=.9)[0]
                    label = torch.from_numpy(V(labels[i])).long()
                    data = torch.cat((pc, label.unsqueeze(1).float()), dim=1).numpy()
                    files_utils.save_np(data, f"{constants.CACHE_ROOT}/seg/{cls.lower()}s/gt/{mesh_name}")
                    # if counter < 20:
                    #     colors = torch.rand(label.max() + 1, 3)[label]
                    #     # files_utils.export_mesh(pc_or, f"{constants.OUT_ROOT}/{cls}/{mesh_name}_pcd_or", colors=colors)
                    #     # files_utils.export_mesh(mesh_or, f"{constants.OUT_ROOT}/{cls}/{mesh_name}_or", colors=colors)
                    #     files_utils.export_mesh(pc, f"{constants.OUT_ROOT}/{cls}/{mesh_name}_pcd", colors=colors)
                    #     files_utils.export_mesh(mesh_wt, f"{constants.OUT_ROOT}/{cls}/{mesh_name}")
                    #     # return
                    # else:
                    #     return
                # counter += 1
                logger.reset_iter()
    logger.stop()




def export_with_mask(mesh, mask, faces_labels, path):
    vs, faces = mesh
    vs_min, vs_max = vs.min(0)[0].unsqueeze(0), vs.max(0)[0].unsqueeze(0)
    # mapper = torch.zeros(vs.shape[0], dtype=torch.int64)
    faces = faces[mask]
    # vs_inds = faces.flatten().unique()
    # mapper[vs_inds] = torch.arange(vs_inds.shape[0])
    # vs = vs[vs_inds]
    # faces = mapper[faces]
    faces_labels = faces_labels[mask]
    # vs = torch.cat((vs, vs_min, vs_max))
    files_utils.export_mesh((vs, faces), path)
    files_utils.export_list(faces_labels.tolist(), f"{path}_faces")


def export_gt(cls: str, names: List[str]):
    def update_mappr(unique_: T, label_: T):
        nonlocal label_mapper, visited_labels
        update_mapper = False
        for k in unique_:
            if k not in visited_labels:
                visited_labels[k] = len(visited_labels) + 1
                update_mapper = True
        if update_mapper:
            label_mapper = torch.zeros(max(list(visited_labels.keys())) + 1, dtype=torch.int64)
            for key, value in visited_labels.items():
                label_mapper[key] = value
        return label_mapper[label_]

    path_metadata = files_utils.collect(f'{constants.PARTNET_ROOT}ins_seg_h5/{cls}/', '.json')
    root_seg = f'{constants.PARTNET_ROOT}sem_seg_h5/{cls}-1/'

    logger = train_utils.Logger().start(len(names))
    counter = 0
    cls_number = get_shapenet_num_class(cls.lower())
    visited_labels = {}
    label_mapper: Optional[T] = None
    for metadata_path in path_metadata:
        metadata = files_utils.load_json(''.join(metadata_path))
        dataset = h5py.File(f'{root_seg}{metadata_path[1]}.h5', "r")
        labels = V(dataset['label_seg'])
        ptc = V(dataset["data"])
        for i in range(ptc.shape[0]):
            mesh_name = metadata[i]["model_id"]
            if mesh_name in names:
                # if not files_utils.is_file(f"{constants.CACHE_ROOT}/seg/{cls.lower()}s/gt/{mesh_name}.npy"):
                out_path = f'{constants.CACHE_ROOT}evaluation/col/chairs/gt/{mesh_name}'
                pc = torch.from_numpy(ptc[i]).float()
                mesh_or = files_utils.load_mesh(f'{constants.Shapenet}{cls_number}/{mesh_name}/models/model_normalized.obj')
                mesh_or = mesh_utils.to_unit_sphere(mesh_or)
                pc[:, 2] = - pc[:, 2]
                pc[:, 0] = - pc[:, 0]
                pc = mesh_utils.to_unit_sphere((pc, None))[0]
                label = torch.from_numpy(V(labels[i])).long()
                label_list = label.unique().tolist()
                label = update_mappr(label_list, label)
                label_list = label.unique().tolist()

                mesh_or = trimesh.remesh.subdivide_to_size(mesh_or[0].numpy(), mesh_or[1].numpy(), 0.1, max_iter=10, return_index=False)
                mesh_or = torch.from_numpy(mesh_or[0]).float(), torch.from_numpy(mesh_or[1]).long()
                vs_labels, faces_labels = mesh_utils.split_by_seg(mesh_or, (pc, label))
                for k in label_list:
                    mask = faces_labels.eq(k)
                    export_with_mask(mesh_or, mask, faces_labels, f"{out_path}_{k}_fg")
                    export_with_mask(mesh_or, ~mask, faces_labels, f"{out_path}_{k}_bg")
                # files_utils.export_mesh(mesh_or, out_path)
                # files_utils.export_list(faces_labels.tolist(), f"{out_path}_faces")
                # if counter < 20:
                #     colors = torch.rand(label.max() + 1, 3)
                #     # files_utils.export_mesh(pc_or, f"{constants.OUT_ROOT}/{cls}/{mesh_name}_pcd_or", colors=colors)
                #     files_utils.export_mesh(mesh_or, f"{constants.OUT_ROOT}/{cls}/{mesh_name}_or", colors=colors[vs_labels])
                #     files_utils.export_mesh(pc, f"{constants.OUT_ROOT}/{cls}/{mesh_name}_pcd", colors=colors[label])
                #     # files_utils.export_mesh(mesh_wt, f"{constants.OUT_ROOT}/{cls}/{mesh_name}")
                #     # return
                # else:
                #     return
                counter += 1
                logger.reset_iter()
    logger.stop()



def sample_folder():
    root = f'{constants.CACHE_ROOT}/evaluation/generation/tables/imnet_tables/'
    paths = files_utils.collect(root, '.ply')[:100]
    loger = train_utils.Logger().start(len(paths))
    for i, path in enumerate(paths):
        out_path = f'{root}pcd_{path[1]}.npy'
        # out_path = f'{constants.CACHE_ROOT}/evaluation/generation/airplanes/imnet/pcd_{i:04d}'
        if not files_utils.is_file(out_path) or True:
            mesh = files_utils.load_mesh(''.join(path))
            if mesh[1].shape[0] > 0:
                mesh = 2 * mesh[0], mesh[1]
                files_utils.export_mesh(mesh,  f'{root}{path[1]}')
                # points = mesh_utils.sample_on_mesh(mesh, 2048, sample_s=mesh_utils.SampleBy.AREAS)[0]
                # files_utils.save_np(points, out_path)
        loger.reset_iter()
    loger.stop()


@models.models_utils.torch_no_grad
def create_im_net_inputs(cls: str, split_name: str):
    res = 256
    empty_min = res ** 3 / 2
    root_shapes = f"{constants.Shapenet_WT}/{cls.lower()}/"
    query_points = mcubes_meshing.MarchingCubesMeshing.get_res_samples(res).transpose(1, 0)[:, :3].numpy().astype(np.float32, order='c')
    # query_points = mcubes_meshing.MarchingCubesMeshing.get_res_samples(256).transpose(1, 0)[:, :3].to(CUDA(0))
    for suffix in ('test', ):
        names = get_split_names(split_name, suffix)
        logger = train_utils.Logger().start(len(names))
        for i, name in enumerate(names):
            out_path = f"{constants.CACHE_ROOT}/im_net_256/{cls}/{name}.npy"
            valid = False
            if files_utils.is_file(out_path):
                labels = files_utils.load_np(out_path)
                valid = labels.sum() > empty_min
            if not valid and files_utils.is_file(f'{root_shapes}{name}.obj'):
                mesh = files_utils.load_mesh(f'{root_shapes}{name}')
                mesh = mesh_utils.to_unit_sphere(mesh, scale=.9)
                label = mesh_utils.get_fast_inside_outside(mesh, query_points)
                if label is not None:
                    files_utils.save_np(label, out_path)
                # pts = query_points[~label]
                # files_utils.export_mesh(pts, f"{constants.OUT_ROOT}/im_net/{name}_pcd")
            # else:
            #     voxels = np.load(f"{constants.CACHE_ROOT}/im_net_256/{cls}/{name}.npy")
            #     pcd = query_points[~voxels]
            #     files_utils.export_mesh(pcd, f"{constants.OUT_ROOT}/occ/{name}")
            #     if i > 10:
            #         break
            logger.reset_iter()
        logger.stop()


if __name__ == '__main__':
    from utils import mesh_closest_point, mcubes_meshing
    names = files_utils.collect(f'{constants.CHECKPOINTS_ROOT}/occ_gmm_chairs_split/occ/', '.obj')
    names = list(map(lambda x: x[1].split('_')[2], names))
    sample_folder()
    # export_points()
#