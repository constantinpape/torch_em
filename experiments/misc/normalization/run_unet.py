# Results (as of Sep. 23, 2024):
# GONUCLEAR
# - (without ignoring empty patches)
# InstanceNorm:           0.2674 (instance segmentation), 0.898 (foreground segmentation)
# InstanceNormTrackStats: 0.4264 (instance segmentation), 0.8766 (foreground segmentation)
# - (ignoring empty patches)
# InstanceNorm:           0.4236 (instance segmentation), 0.899 (foreground segmentation)
# InstanceNormTrackStats: 0.4264 (instance segmentation), 0.8766 (foreground segmentation)

# PLANTSEG (we use without ignoring empty patches only)
# InstanceNorm:           0.5928 (boundary segmentation)
# InstanceNormTrackStats: 0.5532 (boundary segmentation)

# MITOEM (same results with and without ignoring empty patches)
# InstanceNorm:           0.4856 (instance segmentation), 0.9197 (foreground segmentation)
# InstanceNormTrackStats: 0.412 (instance segmentation), 0.9223 (foreground segmentation)


import os
from tqdm import tqdm
from pathlib import Path
from functools import partial

import h5py
import numpy as np
from skimage.segmentation import find_boundaries
from skimage.measure import label as connected_components

import torch

import torch_em
from torch_em.loss import DiceLoss
from torch_em.transform.raw import normalize_percentile
from torch_em.util.prediction import predict_with_halo

from elf.parallel import seeded_watershed
from elf.evaluation import mean_segmentation_accuracy

from common import (
    get_dataloaders, get_model, get_experiment_name, get_test_images, dice_score, _load_image, SAVE_DIR
)


def run_training(name, model, dataset, task, save_root, device):
    n_iterations = int(2.5e4)
    train_loader, val_loader = get_dataloaders(dataset=dataset, task=task)

    trainer = torch_em.default_segmentation_trainer(
        name=name,
        save_root=save_root,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=DiceLoss(),
        metric=DiceLoss(),
        device=device,
        learning_rate=5e-4,
        mixed_precision=True,
        compile_model=False,
        log_image_interval=100,
    )
    trainer.fit(iterations=n_iterations)


def _extract_patchwise_max_intensity(raw, block_shape, halo):
    import nifty.tools as nt
    from torch_em.util.prediction import _load_block

    blocking = nt.blocking([0] * raw.ndim, raw.shape, block_shape)
    n_blocks = blocking.numberOfBlocks
    max_intensities = []
    for block_id in range(n_blocks):
        block = blocking.getBlock(block_id)
        offset = [beg for beg in block.begin]

        inp, _ = _load_block(raw, offset, block_shape, halo)
        max_intensities.append(inp.max())

    return np.max(max_intensities)


def _skip_empty_patches(inp, max_intensity):
    # NOTE: a simple intensity-based approach
    expected_max_intensity = max_intensity / 3
    return inp.max() < expected_max_intensity

    # NOTE: another approach using histograms
    # iflat = inp.flatten()
    # hist, bin_edges = np.histogram(iflat, bins=50)  # calculate the histogram
    # ibins = np.digitize(0.05, bin_edges) - 1  # finding bins correspondng to the desired low intensity threshold
    # icounts = np.sum(hist[:ibins])  # summing up voxel counts below the threshold
    # ipercent = (icounts / iflat.size)  # calculate percentage of voxels found
    # return ipercent > 0.9  # criterion of voxels below threshold exceeding more than 90% considered as an empty block


def run_inference(name, model, norm, dataset, task, save_root, device):
    _do_skip_blocks = False  # skip empty blocks
    _eval_mode = True

    checkpoint = os.path.join(save_root, "checkpoints", name, "best.pt")
    assert os.path.exists(checkpoint), checkpoint

    model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu"))["model_state"])
    model.to(device)

    if _eval_mode:
        model.eval()
    else:
        model.train()

    image_paths, gt_paths = get_test_images(dataset=dataset)

    pred_dir = os.path.join(Path(save_root).parent, "prediction", dataset, norm, task)
    os.makedirs(pred_dir, exist_ok=True)

    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), desc="Predicting", total=len(image_paths)):
        pred_path = os.path.join(pred_dir, f"{Path(image_path).stem}.h5")
        if os.path.exists(pred_path):
            continue

        if dataset == "livecell":
            tile, halo = (384, 384), (64, 64)
        else:
            tile, halo = (16, 384, 384), (8, 64, 64)

        image, _ = _get_per_dataset_inputs(dataset, image_path, gt_path)
        image = normalize_percentile(image)
        if _do_skip_blocks:
            max_intensity = _extract_patchwise_max_intensity(image, tile, halo)
            skip_block = partial(_skip_empty_patches, max_intensity=max_intensity)
        else:
            skip_block = None

        prediction = predict_with_halo(
            input_=image,
            model=model,
            gpu_ids=[device],
            block_shape=tile,
            halo=halo,
            preprocess=None,
            skip_block=skip_block,
        )
        prediction = prediction.squeeze()

        # save outputs
        dname = "segmentation/prediction" if task == "boundaries" else "segmentation/foreground"
        with h5py.File(pred_path, "a") as f:
            f.create_dataset(dname, data=prediction, compression="gzip")


def _get_per_dataset_inputs(dataset, image_path, gt_path):
    if dataset == "livecell":
        image, gt = _load_image(image_path), _load_image(gt_path)
    elif dataset == "gonuclear":
        image, gt = _load_image(image_path, "raw/nuclei"), _load_image(image_path, "labels/nuclei")
    elif dataset == "plantseg":
        image, gt = _load_image(image_path, "raw"), _load_image(image_path, "label")
    elif dataset == "mitoem":
        image, gt = _load_image(image_path, "raw"), _load_image(image_path, "labels")
    else:
        raise ValueError

    return image, gt


def run_evaluation(norm, dataset, task, save_root):
    image_paths, gt_paths = get_test_images(dataset=dataset)

    dsc_list, msa_list, sa50_list = [], [], []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), desc="Evaluating", total=len(image_paths)):
        _, gt = _get_per_dataset_inputs(dataset, image_path, gt_path)

        pred_path = os.path.join(
            Path(save_root).parent, "prediction", dataset, norm, task, f"{Path(image_path).stem}.h5"
        )
        with h5py.File(pred_path, "r+") as f:
            if task == "boundaries" and dataset != "plantseg":
                prediction = f['segmentation/prediction'][:]
                fg, bd = prediction

                if dataset == "livecell":
                    tile, halo = (384, 384), (64, 64)
                else:
                    tile, halo = (16, 384, 384), (8, 64, 64)

                if "instances" in f["segmentation"]:
                    instances = f["segmentation/instances"]
                else:
                    seeds = connected_components((fg - bd) > 0.5)
                    mask = fg > 0.5
                    instances = seeded_watershed(
                        hmap=bd, seeds=seeds, out=np.zeros_like(gt),
                        block_shape=tile, halo=halo, mask=mask, verbose=True,
                    )
                    f.create_dataset("segmentation/foreground", data=fg, compression="gzip")
                    f.create_dataset("segmentation/boundary", data=bd, compression="gzip")
                    f.create_dataset("segmentation/instances", data=instances, compression="gzip")

                msa, sa = mean_segmentation_accuracy(segmentation=instances, groundtruth=gt, return_accuracies=True)
                msa_list.append(msa)
                sa50_list.append(sa[0])

            else:
                if dataset == "plantseg":
                    prediction = f["segmentation/prediction"][:]
                    gt = find_boundaries(gt)
                else:
                    prediction = f["segmentation/foreground"][:]
                    gt = (gt > 0)   # binarise the instances

                score = dice_score(gt=gt, seg=prediction > 0.5)
                assert score > 0 and score <= 1  # HACK: sanity check
                dsc_list.append(score)

    if len(dsc_list) > 0:
        mean_dice = np.mean(dsc_list)
        print(f"Mean dice score for '{dataset}' for the setting: '{task}'-'{norm}: {mean_dice}")
    else:
        mean_msa = np.mean(msa_list)
        mean_sa50 = np.mean(sa50_list)
        print(f"Mean mSA for '{dataset}' for the setting: '{task}'-'{norm}: {mean_msa}, mean SA50: {mean_sa50}")


def run_analysis_per_dataset(dataset, task, save_root):
    exp1_dir = os.path.join(Path(save_root).parent, "prediction", dataset, "InstanceNorm", task)
    exp2_dir = os.path.join(Path(save_root).parent, "prediction", dataset, "InstanceNormTrackStats", task)

    image_paths, gt_paths = get_test_images(dataset=dataset)
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), desc="Analysing", total=len(image_paths)):
        image, gt = _get_per_dataset_inputs(dataset, image_path, gt_path)

        image_id = Path(image_path).stem
        pred_exp1_path = os.path.join(exp1_dir, f"{image_id}.h5")
        pred_exp2_path = os.path.join(exp2_dir, f"{image_id}.h5")

        fg_dname = "segmentation/prediction" if dataset == "plantseg" else "segmentation/foreground"
        with h5py.File(pred_exp1_path, "r") as f1:
            fg_exp1 = f1[fg_dname][:]
            if task == "boundaries" and dataset != "plantseg":
                bd_exp1 = f1["segmentation/boundary"][:]
                instances_exp1 = f1["segmentation/instances"][:]

        with h5py.File(pred_exp2_path, "r") as f2:
            fg_exp2 = f2[fg_dname][:]
            if task == "boundaries" and dataset != "plantseg":
                bd_exp2 = f2["segmentation/boundary"][:]
                instances_exp2 = f2["segmentation/instances"][:]

        import napari
        v = napari.Viewer()
        v.add_image(image, name="Image")
        v.add_labels(gt, name="Ground Truth Labels", visible=False)

        fg_iname = "Boundary" if dataset == "plantseg" else "Foreground"
        v.add_image(fg_exp1, name=f"{fg_iname} (InstanceNorm)", visible=False)
        v.add_image(fg_exp2, name=f"{fg_iname} (InstanceNormTrackStats)", visible=False)

        if task == "boundaries" and dataset != "plantseg":
            v.add_image(bd_exp1, name="Boundary (InstanceNorm)", visible=False)
            v.add_image(bd_exp2, name="Boundary (InstanceNormTrackStats)", visible=False)
            v.add_labels(instances_exp1, name="Instance Segmentation (InstanceNorm)")
            v.add_labels(instances_exp2, name="Instance Segmentation (InstanceNormTrackStats)")
        napari.run()


def main(args):
    phase = args.phase
    dataset = args.dataset
    task = args.task
    norm = args.norm

    if dataset == "plantseg" and task == "binary":
        raise ValueError("The experiment for binary segmentation of 'PlantSeg (Root)' is not implemented.")

    assert task in ["binary", "boundaries"]
    save_root = os.path.join(SAVE_DIR, "models")

    if phase == "evaluate":
        assert norm in ["InstanceNormTrackStats", "InstanceNorm"]
        run_evaluation(norm=norm, dataset=dataset, task=task, save_root=save_root)

    elif phase == "analysis":
        run_analysis_per_dataset(dataset=dataset, task=task, save_root=save_root)

    else:
        assert norm in ["InstanceNormTrackStats", "InstanceNorm"]
        model = get_model(dataset=dataset, task=task, norm=norm)
        name = get_experiment_name(dataset=dataset, task=task, norm=norm, model_choice="unet")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if phase == "train":
            run_training(name=name, model=model, dataset=dataset, task=task, save_root=save_root, device=device)
        elif phase == "predict":
            run_inference(
                name=name, model=model, norm=norm, dataset=dataset, task=task, save_root=save_root, device=device
            )
        else:
            raise ValueError


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, type=str, help="The choice of dataset.")
    parser.add_argument("-p", "--phase", required=True, type=str, help="The mode for running the scripts.")
    parser.add_argument("-t", "--task", required=True, type=str, help="The type of task for segmentation.")
    parser.add_argument("-n", "--norm", type=str, help="The choice of layer normalization.")
    args = parser.parse_args()
    main(args)
