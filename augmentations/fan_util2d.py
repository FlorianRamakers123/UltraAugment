from typing import Sequence, List, Tuple

import torch


def get_fan_mask2d(volume: torch.Tensor, invert: bool = False, return_image : bool = False, smoothing : int = 0):
    """
    Returns a 2D binary mask of the middle slice which indicates which pixels belong to the fan.
    :param return_image:
    :param volume: The tensor representing the volume
    :param invert: Whether to invert the mask. If True, pixels with value 1 do not belong to the fan shape.
    :return: A torch.Tensor object representing the fan mask.
    """
    if volume.ndim == 4:
        volume_ = volume[0] - volume.min()
    else:
        volume_ = volume - volume.min()
    eq0 = (torch.cumsum(volume_, dim=0) == 0) | (_reverse_cumsum(volume_, dim=0) == 0)
    eq1 = (torch.cumsum(volume_, dim=1) == 0) | (_reverse_cumsum(volume_, dim=1) == 0)
    eq2 = (torch.cumsum(volume_, dim=2) == 0) | (_reverse_cumsum(volume_, dim=2) == 0)
    mask = eq0 | eq1 | eq2
    if not invert:
        mask = ~mask

    middle = mask[:,:,mask.shape[-1] // 2]
    if return_image:
        if smoothing > 0:
            slice_i = volume.shape[-1] // 2
            slice = volume[...,slice_i-smoothing:slice_i+smoothing+1].mean(-1).squeeze(0)
        else:
            slice = volume[..., volume.shape[-1] // 2].squeeze(0)
        return middle, slice
    return middle

def _reverse_cumsum(x : torch.Tensor, dim : int | Sequence[int] | None = None) -> torch.Tensor:
    r2lcumsum = x + torch.sum(x, dim=dim, keepdim=True) - torch.cumsum(x, dim=dim)
    return r2lcumsum

def calculate_fan_info2d(fan_mask : torch.Tensor):
    fan_mask = fan_mask.squeeze(0)
    ep = _calculate_extreme_points2d(fan_mask)
    origin = _calculate_origin2d(ep)
    angle = _calculate_angle2d(ep)
    min_dist, max_dist = _calculate_distances2d(fan_mask, origin)
    ell_size, ell_origin, ell_angle = _calculate_ellipse_info2d(ep, origin, min_dist)
    return ep, origin, angle, min_dist, max_dist, ell_size, ell_origin, ell_angle

def _calculate_ellipse_info2d(extreme_points, origin, min_dist):
    (y3,x3), (y1,x1), _, _ = extreme_points
    x2,y2 = origin[1] + min_dist, origin[0]
    oy,ox = origin
    a = torch.sqrt((ox - x2)**2 / 4 + (y2 - y3)**2) + y2 - y3/2.0 - y1/2.0
    c = y3 - y2
    d = -2*x2 + (c / a)**2 * 2 * x2 + 2 * x3
    e = - (c / a)**2
    f = x2**2 - (c / a)**2 * x2**2
    qx = (-d + torch.sqrt(d**2 - 4*e*f)) / (2*e)
    qy = origin[0]
    b = x2 - qx
    l0 = torch.stack((qy - y3,qx - x3), dim=0)
    l1 = torch.tensor([0, b.item()])
    angle = torch.acos(torch.dot(l0,l1) / (torch.norm(l0,p=2) * torch.norm(l1,p=1)))
    return (a,b), (qy,qx), angle

def _calculate_extreme_points2d(mask : torch.Tensor) -> List[torch.Tensor]:
    coords = mask.nonzero().float()
    bl = coords[coords[:,0] == coords[:,0].max()]
    bl = bl[bl[:,1] == bl[:,1].min()].mean(0)
    br = coords[coords[:,0] == coords[:,0].min()]
    br = br[br[:, 1] == br[:, 1].min()].mean(0)
    right = coords[coords[:,0] <= mask.shape[0] // 2]
    left = coords[coords[:,0] > mask.shape[0] // 2]
    if left.numel() == 0:
        pass
    tl = left[left[:,1] == left[:,1].min()]
    tl = tl[tl[:,0] == tl[:,0].max()].mean(0)
    tr = right[right[:,1] == right[:,1].min()]
    tr = tr[tr[:,0] == tr[:,0].min()].mean(0)
    return [tl,tr,bl,br]

def _calculate_origin2d(extreme_points : List[torch.Tensor]) -> torch.Tensor:
    tl,tr,bl,br = extreme_points
    f = tl - bl
    fa = f[0] / f[1]
    fb = bl[0] - fa * bl[1]
    e = tr - br
    ea = e[0] / e[1]
    eb = br[0] - ea * br[1]
    x = (eb - fb) / (fa - ea)
    y = fa * x + fb
    return torch.stack([y,x])

def _calculate_angle2d(extreme_points : List[torch.Tensor]) -> torch.Tensor:
    tl,tr,bl,br = extreme_points
    L = tl - bl
    R = tr - br
    angle = torch.acos(torch.dot(L, R) / (torch.norm(R,p=2) * torch.norm(L,p=2)))
    return angle

def _calculate_distances2d(fan_mask : torch.Tensor, origin : torch.Tensor):
    x0, x1 = torch.meshgrid(torch.arange(0, fan_mask.shape[0]), torch.arange(0, fan_mask.shape[1]))
    dist_to_origin = torch.sqrt((origin[0] - x0)**2 + (origin[1] - x1)**2)
    dists = dist_to_origin[fan_mask.bool()]
    middle = int(round(origin[0].item()))
    scan_line_length = dist_to_origin[middle][fan_mask.bool()[middle]].max() - dists.min()
    return dists.min(), scan_line_length