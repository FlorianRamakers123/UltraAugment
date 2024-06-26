from collections import defaultdict

import torch
from ignite.metrics import Metric
from monai.metrics import compute_dice, compute_hausdorff_distance


class PatientCorrectedDiceScore(Metric):
    required_output_keys = ("y_pred", "y", "patient_ids")

    def __init__(self, keep_channel_dim : bool = True, reduce : bool = False, *args, **kwargs):
        self._buffer = defaultdict(list)
        self.keep_channel_dim = keep_channel_dim
        self.reduce = reduce
        super().__init__(*args, **kwargs)

    def update(self, output):
        y_pred, y, patient_ids = output
        dice_score = compute_dice(torch.stack(y_pred), torch.stack(y), include_background=False, ignore_empty=False)
        if not self.keep_channel_dim:
            dice_score = dice_score.mean(-1)
        for i, pid in enumerate(patient_ids):
            self._buffer[pid].append(dice_score[i])

    def reset(self):
        self._buffer.clear()

    def compute(self):
        pids = [pid for pid in self._buffer]
        patient_dice = torch.stack([torch.stack(self._buffer[pid]).mean(0) for pid in pids])
        if self.reduce:
            return patient_dice.mean(0)
        else:
            return (pids, patient_dice)


class PatientCorrectedHausdorffDistance(Metric):
    required_output_keys = ("y_pred", "y", "patient_ids")

    def __init__(self, keep_channel_dim : bool = True, reduce : bool = False, *args, **kwargs):
        self._buffer = defaultdict(list)
        self.keep_channel_dim = keep_channel_dim
        self.reduce = reduce
        super().__init__(*args, **kwargs)

    def update(self, output):
        y_pred, y, patient_ids = output
        hd_dist = compute_hausdorff_distance(torch.stack(y_pred), torch.stack(y), include_background=False, percentile=95.0)
        if not self.keep_channel_dim:
            hd_dist = hd_dist.mean(-1)
        for i, pid in enumerate(patient_ids):
            self._buffer[pid].append(hd_dist[i])

    def reset(self):
        self._buffer.clear()

    def compute(self):
        pids = [pid for pid in self._buffer]
        patient_hd = torch.stack([torch.stack(self._buffer[pid]).mean(0) for pid in pids])
        if self.reduce:
            return patient_hd.mean(0)
        else:
            return (pids, patient_hd)
