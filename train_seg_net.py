import csv
import glob
import json
import logging
import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.apps import get_logger
from monai.engines import IterationEvents
import sys
from config import ConfigKeys, DEFAULT_CONFIG, TRANSFORMS
from dataloader import plot_segmentation_dataloader, get_dataset_split_folds, get_dataloaders
from train_eval_setup import setup_train_val_test_env

def train_segmentation_network():
    experiment_dir = "train_output/"
    if len(sys.argv) < 2:
        print("Invalid number of cmd line arguments. Please provide a config file.")
        exit()
    config_file = sys.argv[1]

    # Setup directories
    run_dir_prefix = os.path.basename(config_file).rsplit(".", maxsplit=1)[0]
    time_postfix = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    upper_run_dir = os.path.join(experiment_dir, f"{run_dir_prefix}_{time_postfix}")
    os.makedirs(upper_run_dir, exist_ok=True)

    # Read and save config file
    with open(config_file, "r") as f:
        cfg = json.load(f)

    config = deepcopy(DEFAULT_CONFIG)
    config.update(cfg)
    transforms = deepcopy(TRANSFORMS)
    for key in config[ConfigKeys.TRANSFORMS]:
        if key in transforms:
            if "args" in transforms[key]:
                transforms[key]["args"].update(config[ConfigKeys.TRANSFORMS][key].get("args", {}))
            else:
                transforms[key]["args"] = config[ConfigKeys.TRANSFORMS][key].get("args", {})
        else:
            transforms[key] = config[ConfigKeys.TRANSFORMS][key]
    config[ConfigKeys.TRANSFORMS] = transforms

    if config[ConfigKeys.LABEL_FILE] is None:
        print("Label file was not specified in configuration file.")
        exit()

    output_config_file = os.path.join(upper_run_dir, "config.json")
    print(f"Writing configuration to '{output_config_file}'")
    with open(output_config_file, "w+") as file:
        json.dump(config, file)

    folds = []
    data_folds, hold_out_set = get_dataset_split_folds(config)
    for k in range(config[ConfigKeys.K_FOLD_CROSS_VALIDATION]):
        loaders = get_dataloaders(config, data_folds[k])
        lower_run_dir = os.path.join(upper_run_dir, f"fold{k}")
        os.makedirs(lower_run_dir)
        fmt = "[%(levelname)-5.5s][%(asctime)s] %(message)s"
        formatter = logging.Formatter(fmt)
        file_handler = logging.FileHandler(os.path.join(lower_run_dir, "stdout.log"))
        file_handler.setFormatter(formatter)
        l = get_logger(f"train_logger_{k}", fmt=fmt, logger_handler=file_handler)
        trainer, evaluator, tester, summary_writer = setup_train_val_test_env(lower_run_dir, config, l, loaders)
        folds.append((lower_run_dir, trainer, evaluator, tester, l, summary_writer))


    for output_dir, trainer, validator, tester, logger, summary_writer in folds:
        # Plot the data
        logger.info("Plotting datasets...")
        plot_segmentation_dataloader(trainer.data_loader, "image", "mask", "patient_id",
                                     save_path=os.path.join(output_dir, "train_data.png"), ncols=8)
        plot_segmentation_dataloader(validator.data_loader, "image", "mask", num_samples=4, ncols=2,
                                     save_path=os.path.join(output_dir, "val_data.png"))

        # Start training
        logger.info("Starting training process...")
        trainer.run()

        # Test and calculate metrics
        img_seg_pred = []

        @tester.on(IterationEvents.FORWARD_COMPLETED)
        def track_output(engine):
            id = engine.state.output["patient_id"]
            img = engine.state.output["image"]
            seg = engine.state.output["mask"]
            pred = engine.state.output["pred"]
            outputs = torch.softmax(pred, dim=1)
            for i in range(outputs.shape[0]):
                img_seg_pred.append((id[i], img[i], seg[i], outputs[i]))

        logger.info("Running test set...")
        plot_segmentation_dataloader(tester.data_loader, "image", "mask", num_samples=4, ncols=2,
                                     save_path=os.path.join(output_dir, "test_data.png"))
        pt_file = glob.glob(os.path.join(output_dir, "model_net_key_metric=*.pt"))[0]
        logger.info(f"restoring weights from file '{pt_file}'")
        tester.network.load_state_dict(torch.load(pt_file))
        tester.run()
        pids, dice = tester.state.metrics["test_dice"]
        pids2, hd = tester.state.metrics["test_hd"]

        csv_class_eval = os.path.join(output_dir, "test_set_results_dice.csv")
        logger.info(f"Writing results to '{csv_class_eval}'")
        with open(csv_class_eval, "w+", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["patient id"] + [f"dice score {class_name}" for class_name in config[ConfigKeys.CLASS_NAMES][1:]])
            for pid, d in zip(pids, dice):
                writer.writerow([pid] + d.tolist())
        csv_class_eval = os.path.join(output_dir, "test_set_results_hd.csv")
        logger.info(f"Writing results to '{csv_class_eval}'")
        with open(csv_class_eval, "w+", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["patient id"] + [f"hausdorff distance {class_name}" for class_name in
                                  config[ConfigKeys.CLASS_NAMES][1:]])
            for pid2, hd in zip(pids2, hd):
                writer.writerow([pid2] + hd.tolist())

        logger.info("plotting results...")
        for i in range(len(img_seg_pred) // 16 + 1):
            subset = img_seg_pred[i * 16: (i + 1) * 16]
            fig, axs = plt.subplots(4, 4)
            fig.set_figheight(15)
            fig.set_figwidth(15)
            fig.suptitle("Predictions")
            for row in range(4):
                for col in range(4):
                    idx = row * 4 + col
                    if idx >= len(subset):
                        axs[row, col].set_axis_off()
                        continue
                    pid, img, seg, pred = subset[idx]
                    seg = seg.sum(0).cpu().numpy()
                    seg_mask = np.ma.masked_where(seg == 0, seg)
                    pred = (pred[1:].sum(0) > config[ConfigKeys.CONFIDENCE_THRESHOLD]).float().cpu().numpy()
                    pred_mask = np.ma.masked_where(pred == 0, pred)
                    axs[row, col].set_title(str(pid))
                    axs[row, col].imshow(img[0].cpu().numpy(), cmap="gray")
                    axs[row, col].imshow(seg_mask, alpha=0.5, cmap="Greens")
                    axs[row, col].imshow(pred_mask, alpha=0.2, cmap="autumn")
            fig.savefig(os.path.join(output_dir, f"predictions_{i}.png"))
            plt.close(fig)
            plt.show()


if __name__ == "__main__":
    train_segmentation_network()