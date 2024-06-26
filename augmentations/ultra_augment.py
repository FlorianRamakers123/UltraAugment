import warnings
from math import ceil
from typing import Mapping, Hashable, Any, Dict, Tuple, Sequence, Iterable, Generator, cast
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from monai.config import KeysCollection, NdarrayOrTensor
from monai.data import get_track_meta, MetaTensor
from monai.transforms import MapTransform, RandomizableTransform, Zoom, \
    Rand2DElastic, CenterSpatialCrop, create_grid, GridDistortion
from monai.utils import convert_to_tensor, fall_back_tuple, ensure_tuple_rep, InterpolateMode, GridSamplePadMode, \
    GridSampleMode
from skimage.morphology import binary_dilation, binary_erosion, convex_hull_image

from augmentations.fan_util2d import calculate_fan_info2d
from augmentations.fourier2d import polar_fourier_transform2d, inverse_polar_fourier_transform2d
from augmentations.polar_conversion2d import warp_to_polar2d, unwarp_to_cartesian2d

class CalculateFanMask3Dd(MapTransform):
    def __init__(self, keys : KeysCollection, new_key_names : KeysCollection, allow_missing_keys : bool = False):
        super().__init__(keys, allow_missing_keys)
        self.new_key_names = [new_key_names] * len(self.keys) if isinstance(new_key_names, str) else new_key_names

    def _reverse_cumsum(self, x: torch.Tensor, dim: int | Sequence[int] | None = None):
        r2lcumsum = x + torch.sum(x, dim=dim, keepdim=True) - torch.cumsum(x, dim=dim)
        return r2lcumsum

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d : Dict = dict(data)

        for image_key, new_key_name in self.key_iterator(d, self.new_key_names):
            volume = d[image_key]
            eq0 = (torch.cumsum(volume, dim=1) == 0) | (self._reverse_cumsum(volume, dim=1) == 0)
            eq1 = (torch.cumsum(volume, dim=2) == 0) | (self._reverse_cumsum(volume, dim=2) == 0)
            eq2 = (torch.cumsum(volume, dim=3) == 0) | (self._reverse_cumsum(volume, dim=3) == 0)
            mask = ~(eq0 | eq1 | eq2)
            d[new_key_name] = mask
        return d

class CalculateFanMask2Dd(MapTransform):
    def __init__(self, keys: KeysCollection, new_key_names: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.new_key_names = [new_key_names] * len(self.keys) if isinstance(new_key_names, str) else new_key_names
    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict = dict(data)
        for image_key, new_key_name in self.key_iterator(d, self.new_key_names):
            # Extra image processing needed due to little triangle that is present above fan on images from FH/SP dataset
            mask = (~(d[image_key] < 0.005 * d[image_key].max())).float().squeeze(0).numpy()
            for _ in range(20):
                mask = binary_erosion(mask)
            for _ in range(20):
                mask = binary_dilation(mask)
            if mask.shape[0] % 2 != 0:
                mask[mask.shape[0]//2+1:] = np.flip(mask[:mask.shape[0]//2],0)
            else:
                mask[mask.shape[0]//2:] = np.flip(mask[:mask.shape[0]//2], 0)
            mask_c = convex_hull_image(mask.copy())
            y, x = mask_c.nonzero()
            m = x < 60
            mask_c[y[m],x[m]] = mask[y[m], x[m]]
            mask = mask_c
            d[new_key_name] = torch.from_numpy(mask).unsqueeze(0).bool()
        return d


class UltrasoundTransform(MapTransform, RandomizableTransform):
    def __init__(self, keys : KeysCollection, fan_mask_keys : KeysCollection, prob : float, allow_missing_keys : bool = False):
        MapTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.fan_mask_keys = [fan_mask_keys] * len(self.keys) if isinstance(fan_mask_keys, str) else fan_mask_keys

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict = dict(data)
        self.randomize(d)
        d["ULTRA_AUGMENT"] = MetaTensor(torch.tensor(False))
        if not self._do_transform:
            return d
        d["ULTRA_AUGMENT"] = MetaTensor(torch.tensor(True))
        return self.apply_transform(d)

    def apply_transform(self, d: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        return d

class RandSpeckleDistort2Dd(UltrasoundTransform):
    def __init__(self,
                 keys: KeysCollection,
                 fan_mask_keys: KeysCollection,
                 prob: float,
                 phase_offset: float | Tuple[float, float] = torch.pi / 4,
                 factor: Tuple[float, float] = (0.05, 0.05),
                 mixup_alpha: float = None,
                 allow_missing_keys: bool = False):
        super().__init__(keys, fan_mask_keys, prob, allow_missing_keys)
        self.phase_offset = [phase_offset, phase_offset] if isinstance(phase_offset, float) else phase_offset
        self.factor = factor
        self.mixup_alpha = mixup_alpha

    def apply_transform(self, d: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        for image_key, fan_mask_key in self.key_iterator(d, self.fan_mask_keys):
            image = d[image_key].squeeze(0)
            fan_mask = d[fan_mask_key].squeeze(0).bool()
            fft = polar_fourier_transform2d(image, fan_mask)
            mask0 = torch.zeros(fft.shape[0]).bool()
            mask0[int((fft.shape[0] // 2) * self.factor[0]):(fft.shape[0] // 2) + int(ceil((fft.shape[0] // 2) * (1 - self.factor[0])))+1] = True
            slice1 = slice(int(fft.shape[1] * self.factor[1]), None)
            speckle_fft = fft[mask0, slice1]
            idx = torch.from_numpy(self.R.permutation(speckle_fft.nelement()))
            altered_speckle_fft = speckle_fft.view(-1)[idx].view(speckle_fft.size())
            phi_add = self.phase_offset[0] + self.R.random(speckle_fft.shape) * (self.phase_offset[1] - self.phase_offset[0]) # phase_ torch.zeros(altered_speckle_fft.shape, dtype=torch.float32) + phase_offset
            altered_speckle_fft = altered_speckle_fft.abs() * torch.exp(torch.complex(torch.zeros(altered_speckle_fft.shape), torch.from_numpy(phi_add).float() + altered_speckle_fft.angle()))
            lam = self.R.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha is not None else 1.0
            fft[mask0, slice1] = lam * altered_speckle_fft + (1 - lam) * fft[mask0, slice1]
            new_image = inverse_polar_fourier_transform2d(fft, fan_mask)
            d[image_key] = new_image.unsqueeze(0)
        return d

def _calculate_padding2D(fan_mask : torch.Tensor):
    while fan_mask.ndim > 2:
        fan_mask = fan_mask[0]
    coords = fan_mask.nonzero()
    left = coords[:,1].min()
    right = fan_mask.shape[1] - coords[:,1].max() - 1
    top = coords[:,0].min()
    bottom = fan_mask.shape[0] - coords[:,0].max() - 1
    return (left.item(), right.item(), top.item(), bottom.item())

def _ensure_same_padding(image : torch.Tensor, fan_mask : torch.Tensor, target_fan_mask : torch.Tensor):
    lt,rt,tt,bt = _calculate_padding2D(target_fan_mask)
    l,r,t,b = _calculate_padding2D(fan_mask)
    slice0 = slice(t - tt if t > tt else 0, -(b - bt) if b > bt else None)
    slice1 = slice(l - lt if l > lt else 0, -(r - rt) if r > rt else None)
    image = image[...,slice0,slice1]
    padding = (max(0, lt - l), max(0, rt - r), max(0,tt - t), max(0,bt - b))
    image = torch.nn.functional.pad(image, padding)
    return image

class RandElasticFan2Dd(UltrasoundTransform):
    def __init__(self,
                 keys: KeysCollection,
                 fan_mask_keys: KeysCollection,
                 prob: float,
                 spacing : Tuple[float, float] = (100.0, 100.0),
                 magnitude_range : Tuple[float, float] = (2.0, 6.0),
                 padding_mode : str = GridSamplePadMode.ZEROS,
                 mode : str = GridSampleMode.BILINEAR,
                 allow_missing_keys : bool = False):
        super().__init__(keys, fan_mask_keys, prob, allow_missing_keys)
        self.spacing = spacing
        self.magnitude_range = magnitude_range
        self.rand_2d_elastic = Rand2DElastic(spacing=self.spacing, magnitude_range=self.magnitude_range, prob=1)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None):
        self.rand_2d_elastic.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def apply_transform(self, d: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            out: dict[Hashable, NdarrayOrTensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        self.randomize(None)
        device = self.rand_2d_elastic.device
        if device is None and isinstance(d[first_key], torch.Tensor):
            device = d[first_key].device  # type: ignore
            self.rand_2d_elastic.set_device(device)
        if isinstance(d[first_key], MetaTensor) and d[first_key].pending_operations:  # type: ignore
            warnings.warn(f"data['{first_key}'] has pending operations, transform may return incorrect results.")
        sp_size = fall_back_tuple(self.rand_2d_elastic.spatial_size, d[first_key].shape[1:])

        # all the keys share the same random elastic factor
        self.rand_2d_elastic.randomize(sp_size)

        if self._do_transform:
            grid = self.rand_2d_elastic.deform_grid(spatial_size=sp_size)
            grid = self.rand_2d_elastic.rand_affine_grid(grid=grid)
            grid = torch.nn.functional.interpolate(
                recompute_scale_factor=True,
                input=grid.unsqueeze(0),
                scale_factor=ensure_tuple_rep(self.rand_2d_elastic.deform_grid.spacing, 2),
                mode=InterpolateMode.BICUBIC.value,
                align_corners=False,
            )
            grid = CenterSpatialCrop(roi_size=sp_size)(grid[0])
        else:
            grid = cast(torch.Tensor, create_grid(spatial_size=sp_size, device=device, backend="torch"))

        for key, fan_mask_key, mode, padding_mode in self.key_iterator(d, self.fan_mask_keys,  self.mode, self.padding_mode):
            img = d[key]
            warped = torch.zeros_like(img)
            fan_mask = d[fan_mask_key].bool().squeeze(0)
            for c in range(img.shape[0]):
                warped[c] = warp_to_polar2d(img[c].float(), fan_mask)

            deformed = self.rand_2d_elastic.resampler(warped, grid, mode=mode, padding_mode=padding_mode)
            for c in range(img.shape[0]):
                deformed[c] = unwarp_to_cartesian2d(deformed[c], fan_mask)
            d[key] = deformed
        return d

class RandCropFan2Dd(UltrasoundTransform):
    def __init__(self,
                 keys : KeysCollection,
                 fan_mask_keys : KeysCollection,
                 prob : float,
                 fan_angle_factor: float | Tuple[float,float],
                 fan_depth_factor: float | Tuple[float,float],
                 adjust_shape: bool = True,
                 allow_missing_keys : bool = False):
        super().__init__(keys, fan_mask_keys, prob, allow_missing_keys)
        self.fan_angle_factor = fan_angle_factor
        self.fan_depth_factor = fan_depth_factor
        self.adjust_shape = adjust_shape

    def key_iterator(self, data: Mapping[Hashable, Any], *extra_iterables: Iterable | None) -> Generator:
        ex_iters = extra_iterables or [[None] * len(self.keys)]
        _ex_iters: list[Any]
        for key, *_ex_iters in zip(self.keys, *ex_iters):
            if key in data:
                yield (key,) + tuple(_ex_iters) if extra_iterables else key
            elif not self.allow_missing_keys:
                raise KeyError(
                    f"Key `{key}` of transform `{self.__class__.__name__}` was missing in the data"
                    " and allow_missing_keys==False."
                )
        fan_mask_keys = set(self.fan_mask_keys)
        for t in zip(fan_mask_keys, fan_mask_keys):
            yield t

    def apply_transform(self, d: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        fan_angle_factor = self.fan_angle_factor if isinstance(self.fan_angle_factor, float) else self.fan_angle_factor[0] + self.R.random() * (self.fan_angle_factor[1] - self.fan_angle_factor[0])
        fan_depth_factor = self.fan_depth_factor if isinstance(self.fan_depth_factor, float) else self.fan_depth_factor[0] + self.R.random() * (self.fan_depth_factor[1] - self.fan_depth_factor[0])
        for image_key, fan_mask_key in self.key_iterator(d, self.fan_mask_keys):
            image = d[image_key].clone()
            buffer = d[image_key].clone()
            fan_mask = d[fan_mask_key].squeeze(0)
            for c in range(image.shape[0]):
                warped = warp_to_polar2d(image[c].float(), fan_mask)
                warped_fan_mask = warp_to_polar2d(fan_mask.float().clone(), fan_mask)
                y, x = torch.meshgrid(torch.arange(fan_mask.shape[0]), torch.arange(fan_mask.shape[1]))
                _, origin, angle, min_dist, scan_line_length, (a, b), (qy, qx), ellipse_angle = calculate_fan_info2d(fan_mask)
                theta_spacing = angle / fan_mask.shape[0]
                norm = torch.sqrt((origin[0] - y) ** 2 + (origin[1] - x) ** 2)
                theta = torch.pi / 2.0 - torch.acos((y - origin[0]) / norm) + angle / 2
                theta_small = theta / angle * ellipse_angle - ellipse_angle / 2
                theta_idx = theta / theta_spacing
                x_small = qx + b * torch.cos(theta_small)
                y_small = qy + a * torch.sin(theta_small)
                min_dist = torch.sqrt((x_small - origin[1]) ** 2 + (y_small - origin[0]) ** 2)
                r_spacing = (scan_line_length) / (fan_mask.shape[1])
                r = (torch.sqrt((origin[0] - y) ** 2 + (origin[1] - x) ** 2) - min_dist) / r_spacing
                grid = torch.cat((theta_idx.unsqueeze(-1), r.unsqueeze(-1)), dim=-1)
                grid[:, :, 0] = (grid[:, :, 0] / fan_mask.shape[0] - 0.5) * 2
                grid[:, :, 1] = (grid[:, :, 1] / fan_mask.shape[1] - 0.5) * 2
                fan_angle = angle * fan_angle_factor
                theta_offset = int(((angle - fan_angle) / theta_spacing).item())
                if theta_offset >= 2:
                    warped[:theta_offset // 2] = warped.min()
                    warped[-theta_offset // 2:] = warped.min()
                    warped_fan_mask[:theta_offset // 2] = warped_fan_mask.min()
                    warped_fan_mask[-theta_offset // 2:] = warped_fan_mask.min()
                r_offset = int(((1 - fan_depth_factor) * scan_line_length / r_spacing).item())
                if r_offset > 0:
                    warped[:, -r_offset:] = warped.min()
                    warped_fan_mask[:, -r_offset:] = warped_fan_mask.min()
                grid = torch.flip(grid, dims=[2])
                image_bc = warped.unsqueeze(0).unsqueeze(0)
                fan_mask_bc = warped_fan_mask.unsqueeze(0).unsqueeze(0)
                grid_bc = grid.unsqueeze(0)
                unwarped_img = torch.nn.functional.grid_sample(image_bc, grid_bc).squeeze(0).squeeze(0)
                unwarped_fan_mask = torch.nn.functional.grid_sample(fan_mask_bc, grid_bc).squeeze(0).squeeze(0)
                if self.adjust_shape:
                    result = _ensure_same_padding(unwarped_img, unwarped_fan_mask, fan_mask)
                else:
                    result = unwarped_img
                if image.shape[0] > 1:
                    result = result.round()
                buffer[c] = result
            d[image_key] = buffer
        return d
class RandSimulateReverberationAlternative2Dd(UltrasoundTransform):
    def __init__(self,
                 keys : KeysCollection,
                 fan_mask_keys : KeysCollection,
                 prob : float,
                 min_bounds: Tuple[float, float, float, float],
                 max_bounds: Tuple[float, float, float, float] | None = None,
                 num_reverberations : int | Tuple[int, int] = 3,
                 shadow_strength : Tuple[float,float] | float = 0.9,
                 shadow_depth : Tuple[float,float] | float = 0.5,
                 allow_missing_keys : bool = False):
        super().__init__(keys, fan_mask_keys, prob, allow_missing_keys)
        self.shadow_strength = shadow_strength
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.num_reverberations = num_reverberations
        self.shadow_depth = shadow_depth

    def apply_transform(self, d: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        bounds = self.min_bounds if self.max_bounds is None else tuple(min_val + self.R.random() * (max_val - min_val) for min_val, max_val in zip(self.min_bounds, self.max_bounds))
        shadow_strength = self.shadow_strength if isinstance(self.shadow_strength, float) else (self.shadow_strength[0] + self.R.random() * (self.shadow_strength[1] - self.shadow_strength[0]))
        shadow_depth = self.shadow_depth if isinstance(self.shadow_depth, float) else (self.shadow_depth[0] + self.R.random() * (self.shadow_depth[1] - self.shadow_depth[0]))
        num_reverberations = self.num_reverberations if isinstance(self.num_reverberations, int) else self.R.randint(self.num_reverberations[0], self.num_reverberations[1] + 1)
        for image_key, fan_mask_key in self.key_iterator(d, self.fan_mask_keys):
            image = d[image_key].squeeze(0).clone()
            fan_mask = d[fan_mask_key].bool().squeeze(0).clone()
            warped = warp_to_polar2d(image, fan_mask.clone())
            h, w = warped.shape
            px, py = int(bounds[0] * w), int(bounds[1] * h),
            pw, ph = min(int(bounds[2] * w), int((1 - bounds[0]) * w)), min(int(bounds[3] * h), int((1 - bounds[1]) * h))
            patch = torch.zeros_like(image)
            for i in range(num_reverberations):
                patch[py:py+ph,px+i*pw:px+(i+1)*pw] = warped[py:py+ph, px:px+pw].clone()
            smoothing_mask = torch.zeros_like(image)
            y, x = torch.meshgrid(torch.linspace(0.0,1.0,ph), torch.linspace(0.0, 1.0, pw * num_reverberations))
            smoothing_mask[py:py+ph,px:px+num_reverberations*pw] = 0.5 * torch.sin(2 * torch.pi * num_reverberations * x) + 0.5
            blur = torch.zeros_like(image)
            m = min(ph // 2 - 1, 2)
            blur[py + m:py + ph - m, px:px + (num_reverberations * pw) - m] = 1.0
            blur = torchvision.transforms.functional.gaussian_blur(blur.unsqueeze(0),[11,11]).squeeze(0)
            blur /= blur.max()
            smoothing_mask *= blur
            warped = smoothing_mask * patch + (1 - smoothing_mask) * warped
            sd = int(round(shadow_depth * w))
            blur = torch.zeros_like(image)
            m = min(sd-1,20,ph//2-1)
            blur[py+m:py+ph-m, px+(num_reverberations*pw):px+(num_reverberations*pw)+sd-m] = 1.0
            blur = torchvision.transforms.functional.gaussian_blur(blur.unsqueeze(0), [51, 51], 25).squeeze(0)
            if blur.max() != 0:
                blur /= blur.max()
            else:
                print("WARNING: blur.max() was zero so reverberation simulated withouth shadow")
            warped -= shadow_strength * blur * warped
            unwarped = unwarp_to_cartesian2d(warped, fan_mask)
            d[image_key] = unwarped.unsqueeze(0)
        return d

class RandSmoothScaleShadow2Dd(UltrasoundTransform):
    def __init__(self,
                 keys : KeysCollection,
                 fan_mask_keys : KeysCollection,
                 prob : float,
                 shadow_factor : float | Tuple[float,float],
                 allow_missing_keys : bool = False):
        super().__init__(keys, fan_mask_keys, prob, allow_missing_keys)
        self.shadow_factor = shadow_factor
    def apply_transform(self, d: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        shadow_factor = self.shadow_factor if isinstance(self.shadow_factor, float) else self.shadow_factor[0] + self.R.random() * (self.shadow_factor[1] - self.shadow_factor[0])
        for image_key, fan_mask_key in self.key_iterator(d, self.fan_mask_keys):
            image = d[image_key].squeeze(0).clone()
            fan_mask = d[fan_mask_key].bool().squeeze(0).clone()
            warped = warp_to_polar2d(image, fan_mask.clone())
            rescaled = 255.0 * (warped - warped.min()) / (warped.max() - warped.min())
            threshold, _ = cv2.threshold(rescaled.numpy().astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = 0.5 - 0.5 * np.cos(np.pi * rescaled / threshold)
            mask[rescaled > threshold] = 1
            mask = torch.from_numpy(1 - mask)
            unwarped_mask = unwarp_to_cartesian2d(mask, fan_mask.clone())
            unwarped_mask[~fan_mask] = 0.0
            shadow_energy = (image * unwarped_mask * fan_mask).sum()
            other_energy = ((1 - unwarped_mask) * fan_mask * image).sum()
            energy_offset = shadow_factor * shadow_energy
            factor = (energy_offset + other_energy) / other_energy
            image = image * mask * shadow_factor + (1 - mask) * image * factor
            d[image_key] = image.unsqueeze(0)
        return d


class RandMultiplicativeNoise2Dd(UltrasoundTransform):
    def __init__(self,
                 keys : KeysCollection,
                 fan_mask_keys : KeysCollection,
                 prob : float,
                 shape : float | Tuple[float,float],
                 scale : float | Tuple[float,float],
                 allow_missing_keys : bool = False):
        super().__init__(keys, fan_mask_keys, prob, allow_missing_keys)
        self.shape = shape
        self.scale = scale
    def apply_transform(self, d: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        shape = self.shape if isinstance(self.shape, float) else self.shape[0] + self.R.random() * (self.shape[1] - self.shape[0])
        scale = self.scale if isinstance(self.scale, float) else self.scale[0] + self.R.random() * (self.scale[1] - self.scale[0])
        for image_key, fan_mask_key in self.key_iterator(d, self.fan_mask_keys):
            image = d[image_key].squeeze(0)
            fan_mask = d[fan_mask_key].squeeze(0)
            warped = warp_to_polar2d(image, fan_mask.clone())
            gamma_noise = torch.from_numpy(self.R.gamma(shape, scale, size=warped.shape))
            warped *= gamma_noise / gamma_noise.mean()
            unwarped_image = unwarp_to_cartesian2d(warped, fan_mask.clone())
            d[image_key] = unwarped_image.unsqueeze(0)
        return d