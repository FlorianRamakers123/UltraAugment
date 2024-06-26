import torch
from augmentations.polar_conversion2d import warp_to_polar2d, unwarp_to_cartesian2d

def polar_fourier_transform2d(image : torch.Tensor, fan_mask : torch.Tensor) -> torch.Tensor:
    warped_img = warp_to_polar2d(image, fan_mask)
    fft = torch.fft.rfft2(warped_img)
    return fft

def inverse_polar_fourier_transform2d(fft : torch.Tensor, fan_mask : torch.Tensor) -> torch.Tensor:
    warped_img = torch.fft.irfft2(fft)
    unwarped_img = unwarp_to_cartesian2d(warped_img, fan_mask)
    return unwarped_img
