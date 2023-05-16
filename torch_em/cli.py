import argparse
import json
import multiprocessing
import uuid

import imageio.v3 as imageio
import torch
import torch_em
from elf.io import open_file
from torch.utils.data import random_split
from torch_em.model.unet import AnisotropicUNet, UNet2d, UNet3d
from torch_em.util.prediction import predict_with_halo, predict_with_padding


#
# CLI for training
#


def _get_training_parser(description):
    parser = argparse.ArgumentParser(description=description)

    # paths and keys for the data
    # training inputs and labels are required
    parser.add_argument("-i", "--training_inputs", help="", required=True, type=str, nargs="+")
    parser.add_argument("-l",  "--training_labels", help="", required=True, type=str, nargs="+")
    parser.add_argument("-k", "--training_input_key", help="")
    parser.add_argument("--training_label_key", help="")
    # val inputs and labels are optional; if not given we split off parts of the training data
    parser.add_argument("--validation_inputs", help="", type=str, nargs="+")
    parser.add_argument("--validation_labels", help="", type=str, nargs="+")
    parser.add_argument("--validation_input_key", help="")
    parser.add_argument("--validation_label_key", help="")

    # other options
    parser.add_argument("-b", "--batch_size", type=int, help="", required=True)
    parser.add_argument("-p", "--patch_shape", type=int, nargs="+", help="", required=True)
    parser.add_argument("-n", "--n_iterations", type=int, default=25000, help="")
    parser.add_argument("-m", "--label_mode", help="")
    parser.add_argument("--name", help="")
    parser.add_argument("--train_fraction", type=float, default=0.8, help="")

    return parser


# TODO provide an option to over-ride the offsets, e.g. via filepath to a json?
def _get_offsets(ndim, scale_factors):
    if ndim == 2:
        offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3], [-9, 0], [0, -9], [-27, 0], [0, -27]]
    elif ndim == 3 and scale_factors is None:
        offsets = [
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [-3, 0, 0], [0, -3, 0], [0, 0, -3],
            [-9, 0, 0], [0, -9, 0], [0, 0, -9],
            [-27, 0, 0], [0, -27, 0], [0, 0, -27],
        ]
    else:
        offsets = [
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [-2, 0, 0], [0, -3, 0], [0, 0, -3],
            [-3, 0, 0], [0, -9, 0], [0, 0, -9],
            [-4, 0, 0], [0, -27, 0], [0, 0, -27],
        ]
    return offsets


def _get_loader(input_paths, input_key, label_paths, label_key, args, ndim, perform_split=False):
    label_transform, label_transform2 = None, None

    # figure out the label transformations
    label_modes = (
        "affinties", "affinities_and_foreground",
        "boundaries", "boundaries_and_foreground",
        "foreground",
    )
    if args.label_mode is None:
        pass
    elif args.label_mode == "affinities":
        offsets = _get_offsets(ndim, args.scale_factors)
        label_transform = torch_em.transform.label.AffinityTransform(
            offsets=offsets, add_binary_target=False, add_mask=True,
        )
    elif args.label_mode == "affinities_and_foreground":
        label_transform = torch_em.transform.label.AffinityTransform(
            offsets=_get_offsets(ndim, args.scale_factors), add_binary_target=True, add_mask=True,
        )
    elif args.label_mode == "boundaries":
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=False)
    elif args.label_mode == "boundaries_and_foreground":
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)
    elif args.label_mode == "foreground":
        label_transform = torch_em.transform.label.labels_to_binary
    else:
        raise ValueError(f"Unknown label mode {args.label_model}, expect one of {label_modes}")

    # validate the patch shape
    patch_shape = args.patch_shape
    if ndim == 2:
        if len(patch_shape) != 2 and patch_shape[0] != 1:
            raise ValueError(f"Invalid patch_shape {patch_shape} for 2d data.")
    elif ndim == 3:
        if len(patch_shape) != 3:
            raise ValueError(f"Invalid patch_shape {patch_shape} for 3d data.")
    else:
        raise RuntimeError(f"Invalid ndim: {ndim}")

    # TODO figure out if with channels
    ds = torch_em.default_segmentation_dataset(
        input_paths, input_key, label_paths, label_key,
        patch_shape=patch_shape, ndim=ndim,
        label_transform=label_transform,
        label_transform2=label_transform2,
    )

    n_cpus = multiprocessing.cpu_count()
    if perform_split:
        fractions = [args.train_fraction, 1.0 - args.train_fraction]
        ds_train, ds_val = random_split(ds, fractions)
        train_loader = torch_em.segmentation.get_data_loader(
            ds_train, batch_size=args.batch_size, shuffle=True, num_workers=n_cpus
        )
        val_loader = torch_em.segmentation.get_data_loader(
            ds_val, batch_size=args.batch_size, shuffle=True, num_workers=n_cpus
        )
        return train_loader, val_loader
    else:
        loader = torch_em.segmentation.get_data_loader(
            ds, batch_size=args.batch_size, shuffle=True, num_workers=n_cpus
        )
    return loader


def _get_loaders(args, ndim):
    # if validation data is not passed we split the loader
    if args.validation_inputs is None:
        print("You haven't provided validation data so the validation set will be split off the input data.")
        print(f"A fraction of {args.train_fraction} will be used for training and {1 - args.train_fraction} for val.")
        train_loader, val_loader = _get_loader(
            args.training_inputs, args.training_input_key, args.training_labels, args.training_label_key,
            args=args, ndim=ndim, perform_split=True,
        )
    else:
        train_loader = _get_loader(
            args.training_inputs, args.training_input_key, args.training_labels, args.training_label_key,
            args=args, ndim=ndim,
        )
        val_loader = _get_loader(
            args.validation_inputs, args.validation_key, args.validation_labels, args.validation_label_key,
            args=args, ndim=ndim,
        )
    return train_loader, val_loader


def _determine_channels(train_loader, args):
    x, y = next(iter(train_loader))
    in_channels = x.shape[1]
    out_channels = y.shape[1]
    return in_channels, out_channels


def train_2d_unet():
    parser = _get_training_parser("Train a 2D UNet.")
    args = parser.parse_args()

    train_loader, val_loader = _get_loaders(args, ndim=2)
    # TODO more unet settings
    # create the 2d unet
    in_channels, out_channels = _determine_channels(train_loader, args)
    model = UNet2d(in_channels, out_channels)

    if "affinities" in args.label_mode:
        loss = torch_em.loss.LossWrapper(
            torch_em.loss.DiceLoss(),
            transform=torch_em.loss.ApplyAndRemoveMask(masking_method="multiply")
        )
    else:
        loss = torch_em.loss.DiceLoss()

    # generate a random id for the training
    name = f"2d-unet-training-{uuid.uuid1()}" if args.name is None else args.name
    print("Start 2d unet training for", name)
    trainer = torch_em.default_segmentation_trainer(
        name=name, model=model, train_loader=train_loader, val_loader=val_loader,
        loss=loss, metric=loss, compile_model=False,
    )
    trainer.fit(args.n_iterations)


def train_3d_unet():
    parser = _get_training_parser("Train a 3D UNet.")
    parser.add_argument("-s", "--scale_factors", type=str, help="json encoded")
    args = parser.parse_args()

    scale_factors = None if args.scale_factors is None else json.loads(args.scale_factors)
    train_loader, val_loader = _get_loaders(args, ndim=3)

    # TODO more unet settings
    # create the 3d unet
    in_channels, out_channels = _determine_channels(train_loader, args)
    if scale_factors is None:
        model = UNet3d(in_channels, out_channels)
    else:
        model = AnisotropicUNet(in_channels, out_channels, scale_factors)

    if "affinities" in args.label_mode:
        loss = torch_em.loss.LossWrapper(
            torch_em.loss.DiceLoss(),
            transform=torch_em.loss.ApplyAndRemoveMask(masking_method="multiply")
        )
    else:
        loss = torch_em.loss.DiceLoss()

    # generate a random id for the training
    name = f"3d-unet-training-{uuid.uuid1()}" if args.name is None else args.name
    print("Start 3d unet training for", name)
    trainer = torch_em.default_segmentation_trainer(
        name=name, model=model, train_loader=train_loader, val_loader=val_loader,
        loss=loss, metric=loss, compile_model=False,
    )
    trainer.fit(args.n_iterations)


#
# CLI for prediction
#


def _get_prediction_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c", "--checkpoint", required=True, help="")
    parser.add_argument("-i", "--input_path", required=True, help="")
    parser.add_argument("-k", "--input_key", help="")
    parser.add_argument("-o", "--output_path", required=True, help="")
    parser.add_argument("--output_key", help="")
    parser.add_argument("--chunks", nargs="+", type=int, help="")
    parser.add_argument("--compression", help="")
    return parser


def _prediction(args, predict, device):
    model = torch_em.util.get_trainer(args.checkpoint, device=device).model

    if args.input_key is None:
        input_ = imageio.imread(args.input_path)
        pred = predict(model, input_)
    else:
        with open_file(args.input_path, "r") as f:
            input_ = f[args.input_key]
            pred = predict(model, input_)

    output_key = args.output_key
    if output_key is None:
        imageio.imwrite(args.output_path, pred)
    else:
        kwargs = {}
        if args.chunks is not None:
            assert len(args.chunks) == pred.ndim
            kwargs["chunks"] = args.chunks
        if args.compression is not None:
            kwargs["compression"] = args.compression
        with open_file(args.output_path, "a") as f:
            ds = f.require_dataset(
                output_key, shape=pred.shape, dtype=pred.dtype, **kwargs
            )
            ds.n_threads = multiprocessing.cpu_count()
            ds[:] = pred


def predict():
    parser = _get_prediction_parser("Run prediction (with padding if necessary).")
    parser.add_argument("--min_divisible", nargs="+", type=int, help="")
    parser.add_argument("-d", "--device", help="")
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # TODO enable prediction with channels
    def predict(model, input_):
        if args.min_divisible is None:
            with torch.no_grad():
                input_ = torch.from_numpy(input_[:][None, None]).to(device)
                pred = model(input_)
            pred = pred.cpu().numpy().squeeze()
        else:
            pred = predict_with_padding(input_[:], model, args.min_divisible, device)
        return pred

    _prediction(args, predict, device)


def _pred_2d(model, input_):
    assert input_.shape[2] == 1
    pred = model(input_[:, :, 0])
    return pred[:, :, None]


def predict_with_tiling():
    parser = _get_prediction_parser("Run prediction over tiled input.")
    parser.add_argument("-b", "--block_shape", nargs="+", required=True, type=int, help="")
    parser.add_argument("--halo", nargs="+", type=int, help="")
    parser.add_argument("-d", "--devices", nargs="+", help="")
    args = parser.parse_args()

    block_shape = args.block_shape
    if args.halo is None:
        halo = [0] * len(block_shape)
    else:
        halo = args.halo
    assert len(halo) == len(block_shape)

    if args.devices is None:
        devices = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    else:
        devices = args.devices

    # if the block shape is singleton in the first axis we assume that this is a 2d model
    if block_shape[0] == 1:
        pred_function = _pred_2d
    else:
        pred_function = None

    # TODO enable prediction with channels
    def predict(model, input_):
        pred = predict_with_halo(
            input_, model, gpu_ids=devices, block_shape=block_shape, halo=halo, prediction_function=pred_function
        )
        return pred

    _prediction(args, predict, devices[0])
