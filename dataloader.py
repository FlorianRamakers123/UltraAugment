import warnings
from collections import defaultdict
from math import ceil
from pydoc import locate
from typing import Dict, Any
import numpy as np
import torch
import csv
from monai.data.utils import list_data_collate
from monai.data import CacheDataset
from torch.utils.data import DataLoader
from monai.transforms import Compose, OneOf
from config import ConfigKeys
from copy import deepcopy
from typing import Union, Sequence, Tuple, Callable, Optional, List
from einops import rearrange
from monai.config import KeysCollection
from monai.data import Dataset
from monai.transforms import apply_transform

class MixupCutMixDataset(Dataset):

    def __init__(self,
                 data : Dataset,
                 transform : Callable,
                 label_keys: KeysCollection,
                 fan_mask_keys : KeysCollection,
                 image_keys: KeysCollection,
                 do_fan_shape_correction: Union[bool,List[bool]],
                 prob : Tuple[float,float] | Tuple[float, float,float],
                 alpha_mixup : Union[float,Tuple[float, float]],
                 alpha_cutmix : Union[float,Tuple[float, float]],
                 kappa_cutmix : float,
                 pos_range : Tuple[Union[float,Tuple[float, float]]],
                 size_range : Optional[Tuple[Union[float,Tuple[float, float]]]] = None,
                 alpha_patchmix: Union[float, Tuple[float, float]] = 0.2,
                 patch_size : int = 8,
                 preceding : bool = False,
                 seed : int = 1234):
        super().__init__(data, transform)
        self.label_keys = [label_keys] if isinstance(label_keys, str) else label_keys
        self.image_keys = [image_keys] if isinstance(image_keys, str) else image_keys
        self.fan_mask_keys = [fan_mask_keys] if isinstance(fan_mask_keys, str) else fan_mask_keys
        self.do_fan_shape_correction = [do_fan_shape_correction] * len(image_keys) if isinstance(do_fan_shape_correction, bool) else do_fan_shape_correction
        self.prob = np.cumsum(list(prob) + (3 - len(prob)) * [prob[-1]])
        self.preceding = preceding
        self.alpha_mixup = alpha_mixup
        self.alpha_cutmix = alpha_cutmix
        self.alpha_patchmix = alpha_patchmix
        self.kappa_cutmix = kappa_cutmix
        self.pos_range = pos_range
        self.size_range = size_range
        self.patch_size = patch_size
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx):
        other_idx = idx
        while other_idx == idx:
            other_idx = self.rng.integers(low=0, high=len(self))
        d1 = deepcopy(self.data[idx])
        d2 = self.data[other_idx]
        r = self.rng.random()

        if r < self.prob[0] and not d1.get("ULTRA_AUGMENT",False):
            a, b = (self.alpha_mixup, self.alpha_mixup) if isinstance(self.alpha_mixup, float) else self.alpha_mixup
            lam = self.rng.beta(a,b)
            if lam > 0.5:
                lam = 1.0 - lam
            for image_key, fan_mask_key, dfsc in zip(self.image_keys, self.fan_mask_keys, self.do_fan_shape_correction):
                if image_key not in d1:
                    continue
                combined = lam * d1[image_key] + (1 - lam) * d2[image_key]
                if dfsc:
                    d1[image_key], d1[fan_mask_key] = self.fix_fan_shapes(combined, d1[fan_mask_key], d2[fan_mask_key], d1[image_key].min())
                else:
                    d1[image_key] = combined
            for label_key in self.label_keys:
                if label_key not in d1:
                    continue
                d1[label_key] = lam * d1[label_key] + (1 - lam) * d2[label_key]
        elif self.prob[0] <= r < self.prob[1] and not d1.get("ULTRA_AUGMENT",False):
            a, b = (self.alpha_cutmix, self.alpha_cutmix) if isinstance(self.alpha_cutmix, float) else self.alpha_cutmix
            lam = self.rng.beta(a, b)
            for image_key, fan_mask_key, dfsc in zip(self.image_keys, self.fan_mask_keys, self.do_fan_shape_correction):
                img_h, img_w = d1[image_key].shape[1:]
                sample = lambda rof: rof if isinstance(rof, float) else rof[0] + self.rng.random() * (rof[1] - rof[0])
                x = int(round(sample(self.pos_range[0]) * img_w))
                y = int(round(sample(self.pos_range[1]) * img_h))
                if self.size_range is None:
                    width = int(round(img_w * np.sqrt(1 - lam)))
                    height = int(round(img_h * np.sqrt(1 - lam)))
                else:
                    width = int(round(sample(self.size_range[0]) * img_w))
                    height = int(round(sample(self.size_range[1]) * img_h))
                x = min(x, img_w - width)
                y = min(y, img_h - height)

                combined = d1[image_key].clone()
                combined[:, y:y+height, x:x+width] = d2[image_key][:, y:y+height, x:x+width]

                if dfsc:
                    d1[image_key], d1[fan_mask_key] = self.fix_fan_shapes(combined, d1[fan_mask_key], d2[fan_mask_key], d1[image_key].min())
                else:
                    d1[image_key] = combined
                d1[image_key] = (1 - self.kappa_cutmix) * combined + self.kappa_cutmix * d1[image_key]
            for label_key in self.label_keys:
                d1[label_key] = lam * d1[label_key] + (1 - lam) * d2[label_key]
        elif self.prob[1] <= r < self.prob[2] and not d1.get("ULTRA_AUGMENT",False):
            a, b = (self.alpha_patchmix, self.alpha_patchmix) if isinstance(self.alpha_patchmix, float) else self.alpha_patchmix
            lam = self.rng.beta(a, b)
            if lam > 0.5:
                lam = 1.0 - lam

            for image_key in self.image_keys:
                if image_key not in d1:
                    continue
                x1 = d1[image_key]
                x2 = d2[image_key]
                patch_dims = [x1.shape[d+1] // self.patch_size for d in range(x1.ndim - 1)]
                num_patches = torch.prod(torch.tensor(patch_dims)).item()
                ln = int(round(lam * num_patches))
                if x1.ndim == 4:  # 3D
                    x1_patches = rearrange(x1, "c (h1 p1) (w1 p2) (d1 p3) -> c (h1 w1 d1) p1 p2 p3", h1=patch_dims[0], w1=patch_dims[1], d1=patch_dims[2], p1=self.patch_size, p2=self.patch_size, p3=self.patch_size)
                    x2_patches = rearrange(x2, "c (h1 p1) (w1 p2) (d1 p3) -> c (h1 w1 d1) p1 p2 p3", h1=patch_dims[0], w1=patch_dims[1], d1=patch_dims[2], p1=self.patch_size, p2=self.patch_size, p3=self.patch_size)
                else:  # 2D
                    x1_patches = rearrange(x1, "c (h1 p1) (w1 p2) -> c (h1 w1) p1 p2", h1=patch_dims[0], w1=patch_dims[1], p1=self.patch_size, p2=self.patch_size)
                    x2_patches = rearrange(x2, "c (h1 p1) (w1 p2) -> c (h1 w1) p1 p2", h1=patch_dims[0], w1=patch_dims[1], p1=self.patch_size, p2=self.patch_size)

                mask = np.zeros(num_patches)
                mask[:ln] = 1
                mask = self.rng.permutation(mask)
                mask = (torch.from_numpy(mask) == 0)
                combined = x1_patches
                combined[:, mask] = x2_patches[:, mask]
                if x1.ndim == 4:  # 3D
                    combined = rearrange(combined, "c (h1 w1 d1) p1 p2 p3 -> c (h1 p1) (w1 p2) (d1 p3)", h1=patch_dims[0], w1=patch_dims[1], d1=patch_dims[2])
                else:  # 2D
                    combined = rearrange(combined, "c (h1 w1) p1 p2 -> c (h1 p1) (w1 p2)", h1=patch_dims[0], w1=patch_dims[1])
                d1[image_key] = combined
            for label_key in self.label_keys:
                d1[label_key] = lam * d1[label_key] + (1 - lam) * d2[label_key]

        else:
            return super().__getitem__(idx)
        if self.preceding and self.transform is not None:
            d1 = apply_transform(self.transform, d1)
        return d1

    def fix_fan_shapes(self, target_img : torch.Tensor, fan_mask1 : torch.Tensor, fan_mask2 : torch.Tensor, mask_value : torch.Tensor):
        new_fan_mask = (fan_mask1 >= 0.5) # & (fan_mask2 >= 0.5)
        target_img[~new_fan_mask] = mask_value
        return target_img, new_fan_mask


def _construct_transform_chain(config : Dict[str, Any], chain_dict : Dict[str,str]):
    cases = ["train", "validation", "test"]
    out = { case : defaultdict(list) for case in cases }
    for key, d in chain_dict.items():
        for case in cases:
            chain = d if isinstance(d, str) else d[case]
            for name in chain.split("->"):
                name = name.strip()
                if "|" in name:
                    name = tuple(map(lambda s: s.strip(), name.split("|")))
                if name != "":
                    out[case][name].append(key)
    transforms = { case : [] for case in cases }
    for case in cases:
        for name, keys in out[case].items():
            names = name if isinstance(name, tuple) else [name]
            t = []
            for name in names:
                info = config[ConfigKeys.TRANSFORMS][name]
                class_type = locate(info["class"])
                args = info.get("args", {})
                for k,v in args.items():
                    if isinstance(v,str):
                        if v.startswith("\\$"):
                            args[k] = "$" + v[1:]
                        elif v.startswith("$"):
                            args[k] = config[v[1:].lower()]
                t.append(class_type(keys, **args))
            if len(t) > 1:
                transforms[case].append(OneOf(t))
            else:
                transforms[case].append(t[0])
    return transforms


def partition(data : Sequence, classes : Sequence, patients : Sequence, fractions : Sequence[float]):
    sets = [set() for _ in range(len(fractions))]
    set_sizes = [int(round(frac * len(data))) for frac in fractions[:-1]]
    set_sizes.append(len(data) - sum(set_sizes))

    data_per_class = defaultdict(list)
    for i, c in enumerate(classes):
        data_per_class[c].append(i)

    for c in data_per_class:
        class_set_sizes = [int(round(frac * len(data_per_class[c]))) for frac in fractions[:-1]]
        class_set_sizes.append(len(data_per_class[c]) - sum(class_set_sizes))
        class_sets = [set() for _ in range(len(fractions))]
        if len(data_per_class[c]) < len(class_sets):
            warnings.warn(f"Cannot enforce class balance for class {c} since it has less then {len(class_sets)} samples.")
            class_set_sizes = reversed([1 if i < len(data_per_class[c]) else 0 for i in range(len(class_sets))])
        else:
            class_set_sizes = [max(1,csz) for csz in class_set_sizes]
        data_per_patient = defaultdict(list)
        for i in data_per_class[c]:
            data_per_patient[patients[i]].append(i)
        sorted_patients = sorted(data_per_patient.keys(), key=lambda k: len(data_per_patient[k]), reverse=True)
        for p in sorted_patients:
            added = False
            for k in range(len(fractions)):
                if p in [patients[idx] for idx in sets[k]]:
                    class_sets[k] = class_sets[k].union(set(data_per_patient[p]))
                    added = True
                    break
            if not added:
                set_index = max(list(range(len(fractions))), key=lambda k: class_set_sizes[k] - len(class_sets[k]))
                class_sets[set_index] = class_sets[set_index].union(set(data_per_patient[p]))
        for k in range(len(fractions)):
            sets[k] = sets[k].union(class_sets[k])

    patient_sets = [set([patients[i] for i in s]) for s in sets]
    for i in range(len(patient_sets)):
        for j in range(i+1, len(patient_sets)):
            assert len(patient_sets[i].intersection(patient_sets[j])) == 0, "found patient overlap in datasets partition"

    if any(len(sets[i]) != set_sizes[i] for i in range(len(fractions))):
        warnings.warn(f"Dataset split {tuple(len(s) for s in sets)} does not match intended fractions {set_sizes}.")

    splits = [[data[i] for i in s] for s in sets]
    return splits

def get_dataset_split(config : Dict[str,Any]):
    train_val_test_split = config[ConfigKeys.TRAIN_VAL_TEST_SPLIT]
    train_val_test_split = [1 - sum(train_val_test_split)] + train_val_test_split

    with open(config[ConfigKeys.LABEL_FILE], "r", encoding="ascii", errors="ignore") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=",")
        d = list(reader)
    patient_id_key = config[ConfigKeys.PATIENT_ID_KEY]
    if patient_id_key is None:
        patients = [str(i) for i in range(len(d))]
    else:
        patients = [x[patient_id_key] for x in d]

    if config[ConfigKeys.PARTITION_COLUMN_NAME] is not None:
        train_p = [d[i] for i in d if d[i][config[ConfigKeys.PARTITION_COLUMN_NAME]] == "train"]
        val_p = [d[i] for i in d if d[i][config[ConfigKeys.PARTITION_COLUMN_NAME]] == "val"]
        test_p = [d[i] for i in d if d[i][config[ConfigKeys.PARTITION_COLUMN_NAME]] == "test"]
    else:
        if config[ConfigKeys.CLASS_COLUMN_NAME] is not None:
            idc = dict()
            def get_class_idx(c):
                if c not in idc:
                    idc[c] = len(idc)
                return idc[c]
            classes = [get_class_idx(dx[config[ConfigKeys.CLASS_COLUMN_NAME]]) for dx in d]
        else:
            classes = [0 for _ in d]
        train_p, val_p, test_p = partition(d, classes, patients, train_val_test_split)

    return train_p, val_p, test_p


def get_dataset_split_folds(config : Dict[str,Any]):
    test_fraction = (1.0 - config[ConfigKeys.ENSEMBLE_HOLDOUT_FRACTION]) / config[ConfigKeys.K_FOLD_CROSS_VALIDATION]
    fractions = [test_fraction] * config[ConfigKeys.K_FOLD_CROSS_VALIDATION] + [config[ConfigKeys.ENSEMBLE_HOLDOUT_FRACTION]]

    with open(config[ConfigKeys.LABEL_FILE], "r", encoding="ascii", errors="ignore") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=",")
        d = list(reader)

    patient_id_key = config[ConfigKeys.PATIENT_ID_KEY]
    if patient_id_key is None:
        patients = [str(i) for i in range(len(d))]
    else:
        patients = [x[patient_id_key] for x in d]

    folds = []

    if config[ConfigKeys.PARTITION_COLUMN_NAME] is not None:
        holdout_p = [i for i in d if i[config[ConfigKeys.PARTITION_COLUMN_NAME][0]] == "holdout"]
        for name in config[ConfigKeys.PARTITION_COLUMN_NAME]:
            train_p = [i for i in d if i[name] == "train"]
            val_p  = [i for i in d if i[name] == "validation"]
            test_p = [i for i in d if i[name] == "test"]
            folds.append((train_p, val_p, test_p))
    else:
        if config[ConfigKeys.CLASS_COLUMN_NAME] is not None:
            idc = dict()
            def get_class_idx(c):
                if c not in idc:
                    idc[c] = len(idc)
                return idc[c]
            classes = [get_class_idx(dx[config[ConfigKeys.CLASS_COLUMN_NAME]]) for dx in d]
        else:
            classes = [0 for _ in d]
        splits = partition(d, classes, patients, fractions)
        print([len(s) for s in splits])
        for k in range(config[ConfigKeys.K_FOLD_CROSS_VALIDATION]):
            test_set = splits[k]
            val_set = splits[k+1 if k < config[ConfigKeys.K_FOLD_CROSS_VALIDATION] - 1 else 0]
            train_set = sum(splits[:k], start=[]) + sum(splits[k+2:-1], start=[])
            folds.append((train_set, val_set, test_set))
        holdout_p = splits[-1]
    return folds, holdout_p

def get_dataloader(config : Dict[str,Any]):
    with open(config[ConfigKeys.LABEL_FILE], "r", encoding="ascii", errors="ignore") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=",")
        d = list(reader)
    pre_transforms = _construct_transform_chain(config, config[ConfigKeys.PRETRANSFORMS])
    ds = CacheDataset(d, num_workers=None, transform=Compose(pre_transforms["test"]).set_random_state(seed=config[ConfigKeys.SEED]))
    dl = DataLoader(ds, batch_size=config[ConfigKeys.BATCH_SIZE], pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate)
    return dl

def get_dataloaders(config : Dict[str,Any], partitions : Sequence[List[Dict[str,Any]]] | None = None):
    if partitions is None:
        train_p, val_p, test_p = get_dataset_split(config)
    else:
        train_p, val_p, test_p = partitions
    pre_transforms = _construct_transform_chain(config, config[ConfigKeys.PRETRANSFORMS])
    augmentation_transforms = _construct_transform_chain(config, config[ConfigKeys.AUGMENTATION_TRANSFORMS])
    train_ds = CacheDataset(train_p, num_workers=8, transform=Compose(pre_transforms["train"]).set_random_state(seed=config[ConfigKeys.SEED]))
    train_ds = MixupCutMixDataset(train_ds, Compose(augmentation_transforms["train"]).set_random_state(seed=config[ConfigKeys.SEED]),
                                  label_keys=config[ConfigKeys.LABEL_KEYS_CUTMIX_MIXUP],
                                  fan_mask_keys=config[ConfigKeys.FAN_MASK_KEYS_CUTMIX_MIXUP],
                                  image_keys=config[ConfigKeys.IMAGE_KEYS_CUTMIX_MIXUP],
                                  prob=config[ConfigKeys.MIXUP_CUTMIX_PROB],
                                  alpha_mixup=config[ConfigKeys.ALPHA_MIXUP],
                                  preceding=config[ConfigKeys.PRECEDING_CUTMIX_MIXUP],
                                  do_fan_shape_correction=config[ConfigKeys.DO_FAN_SHAPE_CORRECTION],
                                  alpha_cutmix=config[ConfigKeys.ALPHA_CUTMIX],
                                  alpha_patchmix=config[ConfigKeys.ALPHA_PATCHMIX],
                                  patch_size=config[ConfigKeys.PATCH_SIZE],
                                  kappa_cutmix=config[ConfigKeys.KAPPA_CUTMIX],
                                  pos_range=config[ConfigKeys.POS_RANGE_CUTMIX],
                                  size_range=config[ConfigKeys.SIZE_RANGE_CUTMIX], seed=config[ConfigKeys.SEED])
    val_ds = CacheDataset(val_p, num_workers=None, transform=Compose(pre_transforms["validation"]).set_random_state(seed=config[ConfigKeys.SEED]))
    test_ds = CacheDataset(test_p, num_workers=None, transform=Compose(pre_transforms["test"]).set_random_state(seed=config[ConfigKeys.SEED]))
    generator = torch.Generator()
    generator.manual_seed(config[ConfigKeys.SEED])
    train_dl = DataLoader(train_ds, shuffle=True, generator=generator, batch_size=config[ConfigKeys.BATCH_SIZE],
                          pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate)
    val_dl = DataLoader(val_ds, batch_size=config[ConfigKeys.BATCH_SIZE], pin_memory=torch.cuda.is_available(),
                        collate_fn=list_data_collate)
    test_dl = DataLoader(test_ds, batch_size=config[ConfigKeys.BATCH_SIZE], pin_memory=torch.cuda.is_available(),
                          collate_fn=list_data_collate)

    return train_dl, val_dl, test_dl

def get_test_dataloader(config: Dict[str, Any], test_p = None):
    if test_p is None:
        _, _, test_p = get_dataset_split(config)

    pre_transforms = _construct_transform_chain(config, config[ConfigKeys.PRETRANSFORMS])
    test_ds = CacheDataset(test_p, num_workers=None, transform=Compose(pre_transforms["test"]).set_random_state(seed=config[ConfigKeys.SEED]))
    test_dl = DataLoader(test_ds, batch_size=config[ConfigKeys.BATCH_SIZE], pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate)

    return test_dl

def plot_segmentation_dataloader(dataloader : DataLoader, image_key: str, segmentation_key : str, patient_id_key : str = None, num_samples : Optional[int] = None, ncols : int = 4, save_path : Optional[str] = None):
    if num_samples is None:
        num_samples = len(dataloader) * dataloader.batch_size
    samples = []
    for i, d in enumerate(dataloader):
        img_batch, seg_batch = d[image_key], d[segmentation_key]
        pid = d[patient_id_key] if patient_id_key is not None else None
        if i >= ceil(num_samples / dataloader.batch_size):
            break
        for k in range(img_batch.shape[0]):
            if pid is None:
                samples.append((img_batch[k], seg_batch[k]))
            else:
                samples.append((img_batch[k], seg_batch[k], pid[k]))
    samples = samples[:num_samples]

    import matplotlib.pyplot as plt
    nrows = int(ceil(num_samples / ncols))
    fig, axs = plt.subplots(nrows, ncols)
    if len(axs.shape) == 1:
        axs = np.expand_dims(axs, axis=0)
    fig.set_figheight(30)
    fig.set_figwidth(45)
    fig.suptitle(image_key)
    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            if idx >= len(samples):
                continue
            if len(samples[idx]) == 3:
                img, label, pid = samples[idx]
            else:
                (img, label), pid = samples[idx], None
            img = _make_image(img)
            if label.shape[0] == 1:
                seg = _make_image(label[0]).numpy()
            else:
                seg = _make_image(label[1:].sum(0)).numpy()
            seg_mask = np.ma.masked_where(seg == 0, seg)
            if pid is not None:
                axs[row, col].set_title(str(pid))
            axs[row, col].imshow(img.numpy(), cmap="gray")
            axs[row, col].imshow(seg_mask, alpha=0.5, cmap="Greens")

    plt.show()
    if save_path is not None:
        fig.savefig(save_path)

def _make_image(tensor : torch.Tensor):
    while len(tensor.shape) > 2:
        tensor = tensor[tensor.shape[0] // 2]
    return tensor