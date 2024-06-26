import torch
from matplotlib import pyplot as plt

from augmentations.fan_util2d import calculate_fan_info2d

def warp_to_polar2d(image : torch.Tensor, fan_mask : torch.Tensor) -> torch.Tensor:
    _, origin, angle, min_dist, scan_line_length, (a, b), (qy, qx), ellipse_angle = calculate_fan_info2d(fan_mask)
    theta, r = torch.meshgrid(torch.linspace(-angle / 2.0, angle / 2.0, fan_mask.shape[0]), torch.linspace(0, scan_line_length, fan_mask.shape[1]))
    theta_small, _ = torch.meshgrid(torch.linspace(-ellipse_angle / 2.0, ellipse_angle / 2.0, fan_mask.shape[0]), torch.linspace(0, scan_line_length, fan_mask.shape[1]))
    x_small = qx + b * torch.cos(theta_small)
    y_small = qy + a * torch.sin(theta_small)
    min_dist = torch.sqrt((x_small - origin[1])**2 + (y_small - origin[0])**2)
    r = min_dist + r.clone()
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    grid = origin + torch.cat((y.unsqueeze(-1), x.unsqueeze(-1)), dim=-1)
    grid[:, :, 0] = (grid[:, :, 0] / fan_mask.shape[0] - 0.5) * 2
    grid[:, :, 1] = (grid[:, :, 1] / fan_mask.shape[1] - 0.5) * 2
    grid = torch.flip(grid, dims=[2])
    image_bc = image.unsqueeze(0).unsqueeze(0)
    grid_bc = grid.unsqueeze(0)
    warped_img = torch.nn.functional.grid_sample(image_bc, grid_bc).squeeze(0).squeeze(0)
    return warped_img

def unwarp_to_cartesian2d(image : torch.Tensor, fan_mask : torch.Tensor):
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
    r = (torch.sqrt((origin[0] - y) ** 2 + (origin[1] - x) ** 2) - min_dist)  / r_spacing
    grid = torch.cat((theta_idx.unsqueeze(-1), r.unsqueeze(-1)), dim=-1)
    grid[:, :, 0] = (grid[:, :, 0] / fan_mask.shape[0] - 0.5) * 2
    grid[:, :, 1] = (grid[:, :, 1] / fan_mask.shape[1] - 0.5) * 2
    grid = torch.flip(grid, dims=[2])
    image_bc = image.unsqueeze(0).unsqueeze(0)
    grid_bc = grid.unsqueeze(0)
    unwarped_img = torch.nn.functional.grid_sample(image_bc, grid_bc).squeeze(0).squeeze(0)
    return unwarped_img