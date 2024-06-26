from typing import Mapping, Hashable, Dict, Any

import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform

class SliceAtCentreOfMassd(MapTransform):
    def __init__(self, keys: KeysCollection, label_keys : KeysCollection, dim : int, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.dim = dim
        self.label_keys = [label_keys] * len(self.keys) if isinstance(label_keys, str) else label_keys

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d : Dict = dict(data)

        for image_key, label_key in self.key_iterator(d, self.label_keys):
            idx = d[label_key].nonzero().float().mean(0).round().long().tolist()
            dim = self.dim if self.dim >= 0 else d[image_key].ndim + self.dim
            img = d[image_key][(slice(None),)*dim + (idx[dim],)]
            d[image_key] = img

        return d

class ToOneHotd(MapTransform):
    def __init__(self, keys: KeysCollection, num_classes: int, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.num_classes = num_classes

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, any]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            one_hot_label = torch.zeros(self.num_classes, *d[key].shape[1:], dtype=d[key].dtype, device=d[key].device)
            if isinstance(d[key], MetaTensor):
                one_hot_label = MetaTensor(one_hot_label, meta=d[key].meta)
            rounded = d[key].round()
            for channel, val in enumerate(rounded.unique()):
                one_hot_label[channel] = (d[key][0] == val).type(d[key].dtype)
            d[key] = one_hot_label
        return d


