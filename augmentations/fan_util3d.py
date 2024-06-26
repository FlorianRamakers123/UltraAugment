from typing import Sequence, List, Tuple

import torch

def get_fan_mask3d(volume: torch.Tensor, invert: bool = False):
    """
    Returns a 3D binary mask wich indicates which voxels belong to the fan.
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
    return mask

def _reverse_cumsum(x : torch.Tensor, dim : int | Sequence[int] | None = None) -> torch.Tensor:
    r2lcumsum = x + torch.sum(x, dim=dim, keepdim=True) - torch.cumsum(x, dim=dim)
    return r2lcumsum

def calculate_fan_info3d(fan_mask : torch.Tensor):
    fan_mask = fan_mask.squeeze(0)
    ep = _calculate_extreme_points3d(fan_mask)
    origin = _calculate_origin3d(ep)
    angles = _calculate_angles3d(ep)
    min_dist, max_dist = _calculate_distances3d(fan_mask, origin)
    ell_size, ell_origin, ell_angle = _calculate_ellipse_info3d(ep, origin, min_dist)
    return ep, origin, angles, min_dist, max_dist, ell_size, ell_origin, ell_angle

def _calculate_ellipse_info3d(extreme_points, origin, min_dist):
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

def _calculate_extreme_points3d(mask : torch.Tensor) -> List[torch.Tensor]:
    coords = mask.nonzero().float()
    ant_coords = coords[coords[:, 2] < mask.shape[2] // 2]
    post_coords = coords[coords[:, 2] >= mask.shape[2] // 2]
    ant_top_coords = ant_coords[ant_coords[:, 1] < mask.shape[1] // 4]
    ant_bottom_coords = ant_coords[ant_coords[:, 1] >= mask.shape[1] // 4]
    post_top_coords = post_coords[post_coords[:, 1] < mask.shape[1] // 4]
    post_bottom_coords = post_coords[post_coords[:, 1] >= mask.shape[1] // 4]

    mbla = ant_bottom_coords[ant_bottom_coords[:, 0] == ant_bottom_coords[:, 0].min()]
    bla = mbla[mbla[:, 1] == mbla[:, 1].min()].mean(0)
    mbra = ant_bottom_coords[ant_bottom_coords[:, 0] == ant_bottom_coords[:, 0].max()]
    bra = mbra[mbra[:, 1] == mbra[:, 1].min()].mean(0)
    mtla = ant_top_coords[ant_top_coords[:, 1] == ant_top_coords[:, 1].min()]
    tla = mtla[mtla[:, 0] == mtla[:, 0].min()].mean(0)
    mtra = ant_top_coords[ant_top_coords[:, 1] == ant_top_coords[:, 1].min()]
    tra = mtra[mtra[:, 0] == mtra[:, 0].max()].mean(0)

    mblp = post_bottom_coords[post_bottom_coords[:, 0] == post_bottom_coords[:, 0].min()]
    blp = mblp[mblp[:, 1] == mblp[:, 1].min()].mean(0)
    mbrp = post_bottom_coords[post_bottom_coords[:, 0] == post_bottom_coords[:, 0].max()]
    brp = mbrp[mbrp[:, 1] == mbrp[:, 1].min()].mean(0)
    mtlp = post_top_coords[post_top_coords[:, 1] == post_top_coords[:, 1].min()]
    tlp = mtlp[mtlp[:, 0] == mtlp[:, 0].min()].mean(0)
    mtrp = post_top_coords[post_top_coords[:, 1] == post_top_coords[:, 1].min()]
    trp = mtrp[mtrp[:, 0] == mtrp[:, 0].max()].mean(0)

    return [tla, tlp, tra, trp, bla, blp, bra, brp]
def _calculate_origin3d(extreme_points : List[torch.Tensor]) -> torch.Tensor:
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

def _calculate_angles3d(extreme_points : List[torch.Tensor]) -> torch.Tensor:
    tl,tr,bl,br = extreme_points
    L = tl - bl
    R = tr - br
    angle = torch.acos(torch.dot(L, R) / (torch.norm(R,p=2) * torch.norm(L,p=2)))
    return angle

def _calculate_distances3d(fan_mask : torch.Tensor, origin : torch.Tensor):
    x0, x1 = torch.meshgrid(torch.arange(0, fan_mask.shape[0]), torch.arange(0, fan_mask.shape[1]))
    dist_to_origin = torch.sqrt((origin[0] - x0)**2 + (origin[1] - x1)**2)
    dists = dist_to_origin[fan_mask.bool()]
    middle = int(round(origin[0].item()))
    scan_line_length = dist_to_origin[middle][fan_mask.bool()[middle]].max() - dists.min()
    return dists.min(), scan_line_length