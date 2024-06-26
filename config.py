import torch


class ConfigKeys:
    CLASS_COLUMN_NAME = "class_column_name"
    PARTITION_COLUMN_NAME = "partition_column_name"
    PATIENT_ID_KEY = "patient_id_key"
    DEVICE = "device"
    BATCH_SIZE = "batch_size"
    TRAIN_VAL_TEST_SPLIT = "train_val_test_split"
    CLASS_NAMES = "class_names"
    NUM_CLASSES = "num_classes"
    SEED = "seed"
    SPATIAL_SIZE = "spatial_size"
    MODEL_INPUT_KEYS = "model_input_keys"
    IS_MULTI_LABEL = "is_multi_label"
    NUM_EPOCHS = "num_epochs"
    TRAIN_METRICS = "train_metrics"
    VALIDATION_METRICS = "validation_metrics"
    SMALLEST_VAL_METRIC = "smallest_val_metric"
    TEST_METRICS = "test_metrics"
    VALIDATION_INTERVAL = "validation_interval"
    VALIDATION_KEY_METRIC = "validation_key_metric"
    TRAINING_CHECKPOINT_INTERVAL = "training_checkpoint_interval"
    ENFORCE_SINGLE_LABEL_PATIENTS = "enforce_single_label_patients"
    NUM_CHANNELS = "num_channels"
    LABEL_FILE = "label_file"
    PRETRANSFORMS = "pretransforms"
    AUGMENTATION_TRANSFORMS = "augmentation_transforms"
    POSTTRANSFORMS = "posttransforms"
    TRANSFORMS = "transforms"
    NETWORK_CLASS = "network_class"
    NETWORK_ARGS = "network_args"
    OPTIMIZER_CLASS = "optimizer_class"
    OPTIMIZER_ARGS = "optimizer_args"
    LOSS_FUNC = "loss_func"
    DEEP_SUPERVISION = "deep_supervision"
    PATIENT_AGGREGATION = "patient_aggregation"
    CONFIDENCE_THRESHOLD = "confidence_treshold"
    MIXUP_CUTMIX_PROB = "mixup_cutmix_prob"
    DO_FAN_SHAPE_CORRECTION = "do_fan_shape_correction"
    ALPHA_MIXUP = "alpha_mixup"
    ALPHA_CUTMIX = "alpha_cutmix"
    ALPHA_PATCHMIX = "alpha_patchmix"
    PATCH_SIZE = "patch_size"
    KAPPA_CUTMIX = "kappa_cutmix"
    POS_RANGE_CUTMIX = "pos_range_cutmix"
    SIZE_RANGE_CUTMIX = "size_range_cutmix"
    FAN_MASK_KEYS_CUTMIX_MIXUP = "fan_mask_keys_cutmix_mixup"
    LABEL_KEYS_CUTMIX_MIXUP = "label_keys_cutmix_mixup"
    IMAGE_KEYS_CUTMIX_MIXUP = "image_keys_cutmix_mixup"
    PRECEDING_CUTMIX_MIXUP = "preceding_cutmix_mixup"
    K_FOLD_CROSS_VALIDATION = "k_fold_cross_validation"
    ENSEMBLE_AGGREGATION = "ensemble_aggregation"
    ENSEMBLE_HOLDOUT_FRACTION = "ensemble_holdout_fraction"
    USE_CHECKPOINTING = "use_checkpointing"
    PRED_OUTPUT_KEYS = "pred_output_keys"

class PatientAggregation:
    MAJORITY_VOTE = "majority_vote"
    AVERAGE_PROBABILITY = "average_probability"

    @staticmethod
    def do_aggregation(agg : str, x : torch.Tensor):
        if agg == PatientAggregation.MAJORITY_VOTE:
            s = torch.mean(x, dim=0)
            s[s != s.max()] = 0.0
            return s
        elif agg == PatientAggregation.AVERAGE_PROBABILITY:
            s = torch.mean(x, dim=0)
            return s
        raise ValueError(f"Unknown patient aggregation method: '{agg}'")


TRANSFORMS = {
    "Identityd": {
        "class": "monai.transforms.Identityd",
    },
    "LoadImaged": {
        "class": "monai.transforms.LoadImaged",
    },
    "ToOneHotd" : {
        "class": "util.transforms_util.ToOneHotd",
        "args": {
            "num_classes": "$NUM_CLASSES"
        }
    },
    "SliceAtCentreOfMassd": {
        "class": "util.transforms_util.SliceAtCentreOfMassd",
        "args": {
            "label_keys": "label",
            "dim": -1
        }
    },
    "EnsureChannelFirstd": {
        "class": "monai.transforms.EnsureChannelFirstd"
    },
    "Orientationd": {
        "class": "monai.transforms.Orientationd",
        "args": {
            "axcodes": "LPS"
        }
    },
    "Spacingd": {
        "class": "monai.transforms.Spacingd",
        "args": {
            "pixdim": [0.25,0.25,0.25],
            "mode": "bilinear"
        }
    },
    "SpatialPadd": {
        "class": "monai.transforms.SpatialPadd",
        "args": {
            "spatial_size": [256,256]
        }
    },
    "Resized": {
        "class": "monai.transforms.Resized",
        "args": {
            "spatial_size": "$SPATIAL_SIZE",
            "size_mode": "longest",
            "mode": "bilinear"
        }
    },
    "ResizeNearestd": {
        "class": "monai.transforms.Resized",
        "args": {
            "spatial_size": "$SPATIAL_SIZE",
            "size_mode": "longest",
            "mode": "nearest"
        }
    },
    "Rotated": {
        "class": "monai.transforms.Rotated",
        "args": {
            "angle": 0.0
        }
    },
    "ScaleIntensityd": {
        "class": "monai.transforms.ScaleIntensityd",
        "args": {
            "minv": 0.0,
            "maxv": 1.0
        }
    },
    "RepeatChanneld": {
        "class": "monai.transforms.RepeatChanneld",
        "args": {
            "repeats": "$NUM_CHANNELS"
        }
    },
    "NormalizeIntensityd": {
        "class": "monai.transforms.NormalizeIntensityd",
        "args": {}
    },
    "EnsureTyped": {
        "class": "monai.transforms.EnsureTyped",
    },
    "AsDiscreted": {
        "class": "monai.transforms.AsDiscreted",
        "args": {
            "threshold": 0.5
        }
    },
    "Activationsd": {
        "class": "monai.transforms.Activationsd",
        "args": {
            "softmax": True
        }
    },
    "RandFlipd": {
        "class": "monai.transforms.RandFlipd",
        "args": {
            "prob": 0.3,
            "spatial_axis": [0,1]
        }
    },
    "RandSimulateLowResolutiond": {
        "class": "monai.transforms.RandSimulateLowResolutiond",
        "args" : {
            "prob": 0.3,
            "zoom_range": [0.5, 1.0]
        }
    },
    "RandGridDistortiond": {
        "class": "monai.transforms.RandGridDistortiond",
        "args": {
            "prob": 0.3,
            "num_cells": 5,
            "distort_limit": [-0.05, 0.05],
            "padding_mode": "zeros"
        }
    },
    "Rand2DElasticd": {
        "class": "monai.transforms.Rand2DElasticd",
        "args": {
            "prob": 0.3,
            "spacing": [60, 60],
            "magnitude_range": [1,3]
        }
    },
    "Rand3DElasticd": {
        "class": "monai.transforms.Rand3DElasticd",
        "args": {
            "prob": 0.0,
            "sigma_range": [0.5, 2.0],
            "magnitude_range": [1,2]
        }
    },
    "RandHistogramShiftd": {
        "class": "monai.transforms.RandHistogramShiftd",
        "args": {
            "prob": 0.3,
            "num_control_points": 3
        }
    },
    "RandAdjustContrastd": {
        "class": "monai.transforms.RandAdjustContrastd",
        "args": {
            "prob": 0.3,
            "gamma": [0.8, 1.5]
        }
    },
    "RandGaussianNoised": {
        "class": "monai.transforms.RandGaussianNoised",
        "args": {
            "prob": 0.3,
            "mean": 0.0,
            "std": 0.4
        }
    },
    "RandGaussianSmoothd": {
        "class": "monai.transforms.RandGaussianSmoothd",
        "args": {
            "prob": 0.3,
            "sigma_x": [0.25, 1.0],
            "sigma_y": [0.25, 1.0],
            "sigma_z": [0.25, 1.0]
        }
    },
    "RandAffined": {
        "class": "monai.transforms.RandAffined",
        "args": {
            "prob": 0.3,
            "rotate_range": [-.2618, 0.2618],
            "padding_mode": "zeros",
        }
    },
    "RandScaleIntensityd": {
        "class": "monai.transforms.RandScaleIntensityd",
        "args": {
            "prob": 0.3,
            "factors": [0.5, 1.5]
        }
    },
    "CalculateFanMask3Dd": {
        "class": "augmentations.ultra_augment.CalculateFanMask3Dd",
        "args": {
            "new_key_names": "fan_mask",
        }
    },
    "CalculateFanMask2Dd": {
        "class": "augmentations.ultra_augment.CalculateFanMask2Dd",
        "args": {
            "new_key_names": "fan_mask",
        }
    },
    "RandSpeckleDistort2Dd": {
        "class": "augmentations.ultra_augment.RandSpeckleDistort2Dd",
        "args": {
            "fan_mask_keys": "fan_mask",
            "prob": 0.3,
            "phase_offset": [-torch.pi, torch.pi],
            "factor": (0.6,0.6),
            "mixup_alpha": None
        }
    },
    "RandCropFan2Dd": {
        "class": "augmentations.ultra_augment.RandCropFan2Dd",
        "args": {
            "fan_mask_keys": "fan_mask",
            "prob" : 0.3,
            "fan_angle_factor" : [0.8, 0.99],
            "fan_depth_factor" : [0.8, 0.99],
            "adjust_shape" : False,
        }
    },
    "RandSimulateReverberationAlternative2Dd": {
        "class": "augmentations.ultra_augment.RandSimulateReverberationAlternative2Dd",
        "args": {
            "fan_mask_keys": "fan_mask",
            "prob": 0.3,
            "min_bounds": (0.0, 0.0, 0.02, 0.15),
            "max_bounds": (0.0, 0.85, 0.04, 0.75),
            "shadow_strength": 0.8
        }
    },
    "RandSmoothScaleShadow2Dd": {
        "class": "augmentations.ultra_augment.RandSmoothScaleShadow2Dd",
        "args": {
            "fan_mask_keys" : "fan_mask",
            "prob" : 0.3,
            "shadow_factor" : [0.5, 0.8]
        }
    },
    "RandElasticFan2Dd": {
        "class": "augmentations.ultra_augment.RandElasticFan2Dd",
        "args": {
            "fan_mask_keys" : "fan_mask",
            "prob" : 0.3,
            "spacing" : [60, 60],
            "magnitude_range" : [1,3]
        }
    },
    "RandMultiplicativeNoise2Dd": {
        "class": "augmentations.ultra_augment.RandMultiplicativeNoise2Dd",
        "args": {
            "fan_mask_keys": "fan_mask",
            "prob": 0.3,
            "shape": [40, 100],
            "scale": 1.0
        }
    }







}

DEFAULT_CONFIG = {
    ConfigKeys.DEVICE: "cuda:0" if torch.cuda.is_available() else "cpu",
    ConfigKeys.BATCH_SIZE: 10,
    ConfigKeys.NUM_EPOCHS: 200,
    ConfigKeys.VALIDATION_KEY_METRIC: None,
    ConfigKeys.VALIDATION_INTERVAL: 5,
    ConfigKeys.TRAIN_VAL_TEST_SPLIT: [0.1, 0.1],
    ConfigKeys.CLASS_NAMES: ["class1", "class2"],
    ConfigKeys.SPATIAL_SIZE: [256,256],
    ConfigKeys.NUM_CLASSES: 2,
    ConfigKeys.IS_MULTI_LABEL: False,
    ConfigKeys.SEED: 1234,
    ConfigKeys.MODEL_INPUT_KEYS: ["image"],
    ConfigKeys.NUM_CHANNELS: 1,
    ConfigKeys.TRAINING_CHECKPOINT_INTERVAL: 100,
    ConfigKeys.ENFORCE_SINGLE_LABEL_PATIENTS: True,
    ConfigKeys.CONFIDENCE_THRESHOLD: 0.5,
    ConfigKeys.LABEL_FILE: None,
    ConfigKeys.PATIENT_ID_KEY: "id",
    ConfigKeys.PARTITION_COLUMN_NAME: None,
    ConfigKeys.SMALLEST_VAL_METRIC: False,
    ConfigKeys.CLASS_COLUMN_NAME: None,
    ConfigKeys.PRETRANSFORMS: None,
    ConfigKeys.AUGMENTATION_TRANSFORMS: None,
    ConfigKeys.POSTTRANSFORMS: {
        "pred": "Activationsd->AsDiscreted"
    },
    ConfigKeys.TRAIN_METRICS: None,
    ConfigKeys.VALIDATION_METRICS: None,
    ConfigKeys.TEST_METRICS: None,
    ConfigKeys.NETWORK_CLASS: None,
    ConfigKeys.NETWORK_ARGS: None,
    ConfigKeys.OPTIMIZER_CLASS: "torch.optim.Adam",
    ConfigKeys.OPTIMIZER_ARGS: { "lr": 0.0001, "weight_decay": 3e-4 },
    ConfigKeys.LOSS_FUNC: [{
        "class": "torch.nn.CrossEntropyLoss",
        "weight": 1.0,
        "deepsupervision": False,
        "input_keys": ["pred", "label"],
        "args": { "label_smoothing": 0.1 }
    }],
    ConfigKeys.DEEP_SUPERVISION: False,
    ConfigKeys.PATIENT_AGGREGATION: PatientAggregation.AVERAGE_PROBABILITY,
    ConfigKeys.IMAGE_KEYS_CUTMIX_MIXUP: None,
    ConfigKeys.LABEL_KEYS_CUTMIX_MIXUP: None,
    ConfigKeys.FAN_MASK_KEYS_CUTMIX_MIXUP: None,
    ConfigKeys.MIXUP_CUTMIX_PROB: [0.25,0.25],
    ConfigKeys.PRECEDING_CUTMIX_MIXUP: True,
    ConfigKeys.DO_FAN_SHAPE_CORRECTION: False,
    ConfigKeys.ALPHA_MIXUP: 1.0,
    ConfigKeys.ALPHA_CUTMIX: 1.0,
    ConfigKeys.KAPPA_CUTMIX: 0.0,
    ConfigKeys.POS_RANGE_CUTMIX: [0.5, 0.5],
    ConfigKeys.SIZE_RANGE_CUTMIX: None,
    ConfigKeys.TRANSFORMS: {},
    ConfigKeys.K_FOLD_CROSS_VALIDATION: 5,
    ConfigKeys.ENSEMBLE_AGGREGATION: "average",
    ConfigKeys.ENSEMBLE_HOLDOUT_FRACTION: 0.0,
    ConfigKeys.USE_CHECKPOINTING: False,
    ConfigKeys.PATCH_SIZE: 8,
    ConfigKeys.ALPHA_PATCHMIX: 0.2,
    ConfigKeys.PRED_OUTPUT_KEYS: "pred"
}
