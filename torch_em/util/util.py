import os
import warnings
from collections import OrderedDict
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch_em
from matplotlib import colors
from numpy.typing import ArrayLike

try:
    from torch._dynamo.eval_frame import OptimizedModule
except ImportError:
    OptimizedModule = None

# torch doesn't support most unsigned types,
# so we map them to their signed equivalent
DTYPE_MAP = {
    np.dtype("uint16"): np.int16,
    np.dtype("uint32"): np.int32,
    np.dtype("uint64"): np.int64
}
"""@private
"""


# This is a fairly brittle way to check if a module is compiled.
# Would be good to find a better solution, ideall something like model.is_compiled().
def is_compiled(model):
    """@private
    """
    if OptimizedModule is None:
        return False
    return isinstance(model, OptimizedModule)


def auto_compile(
    model: torch.nn.Module, compile_model: Optional[Union[str, bool]] = None, default_compile: bool = True
) -> torch.nn.Module:
    """Automatically compile a model for pytorch >= 2.

    Args:
        model: The model.
        compile_model: Whether to comile the model.
            If None, it will not be compiled for torch < 2, and for torch > 2 the behavior
            specificed by 'default_compile' will be used. If a string is given it will be
            intepreted as the compile mode (torch.compile(model, mode=compile_model))
        default_compile: Whether to use the default compilation behavior for torch 2.

    Returns:
        The compiled model.
    """
    torch_major = int(torch.__version__.split(".")[0])

    if compile_model is None:
        if torch_major < 2:
            compile_model = False
        elif is_compiled(model):  # model is already compiled
            compile_model = False
        else:
            compile_model = default_compile

    if compile_model:
        if torch_major < 2:
            raise RuntimeError("Model compilation is only supported for pytorch 2")

        print("Compiling pytorch model ...")
        if isinstance(compile_model, str):
            model = torch.compile(model, mode=compile_model)
        else:
            model = torch.compile(model)

    return model


def ensure_tensor(tensor: Union[torch.Tensor, ArrayLike], dtype: Optional[str] = None) -> torch.Tensor:
    """Ensure that the input is a torch tensor, by converting it if necessary.

    Args:
        tensor: The input object, either a torch tensor or a numpy-array like object.
        dtype: The required data type for the output tensor.

    Returns:
        The input, converted to a torch tensor if necessary.
    """
    if isinstance(tensor, np.ndarray):
        if np.dtype(tensor.dtype) in DTYPE_MAP:
            tensor = tensor.astype(DTYPE_MAP[tensor.dtype])
        # Try to convert the tensor, even if it has wrong byte-order
        try:
            tensor = torch.from_numpy(tensor)
        except ValueError:
            tensor = tensor.view(tensor.dtype.newbyteorder())
            if np.dtype(tensor.dtype) in DTYPE_MAP:
                tensor = tensor.astype(DTYPE_MAP[tensor.dtype])
            tensor = torch.from_numpy(tensor)

    assert torch.is_tensor(tensor), f"Cannot convert {type(tensor)} to torch"
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def ensure_tensor_with_channels(
    tensor: Union[torch.Tensor, ArrayLike], ndim: int, dtype: Optional[str] = None
) -> torch.Tensor:
    """Ensure that the input is a torch tensor of a given dimensionality with channels.

    Args:
        tensor: The input tensor or numpy-array like data.
        ndim: The dimensionality of the output tensor.
        dtype: The data type of the output tensor.

    Returns:
        The input converted to a torch tensor of the requested dimensionality.
    """
    assert ndim in (2, 3, 4), f"{ndim}"
    tensor = ensure_tensor(tensor, dtype)
    if ndim == 2:
        assert tensor.ndim in (2, 3, 4, 5), f"{tensor.ndim}"
        if tensor.ndim == 2:
            tensor = tensor[None]
        elif tensor.ndim == 4:
            assert tensor.shape[0] == 1, f"{tensor.shape}"
            tensor = tensor[0]
        elif tensor.ndim == 5:
            assert tensor.shape[:2] == (1, 1), f"{tensor.shape}"
            tensor = tensor[0, 0]
    elif ndim == 3:
        assert tensor.ndim in (3, 4, 5), f"{tensor.ndim}"
        if tensor.ndim == 3:
            tensor = tensor[None]
        elif tensor.ndim == 5:
            assert tensor.shape[0] == 1, f"{tensor.shape}"
            tensor = tensor[0]
    else:
        assert tensor.ndim in (4, 5), f"{tensor.ndim}"
        if tensor.ndim == 5:
            assert tensor.shape[0] == 1, f"{tensor.shape}"
            tensor = tensor[0]
    return tensor


def ensure_array(array: Union[np.ndarray, torch.Tensor], dtype: str = None) -> np.ndarray:
    """Ensure that the input is a numpy array, by converting it if necessary.

    Args:
        array: The input torch tensor or numpy array.
        dtype: The dtype of the ouptut array.

    Returns:
        The input converted to a numpy array if necessary.
    """
    if torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    assert isinstance(array, np.ndarray), f"Cannot convert {type(array)} to numpy"
    if dtype is not None:
        array = np.require(array, dtype=dtype)
    return array


def ensure_spatial_array(array: Union[np.ndarray, torch.Tensor], ndim: int, dtype: str = None) -> np.ndarray:
    """Ensure that the input is a numpy array of a given dimensionality.

    Args:
        array: The input numpy array or torch tensor.
        ndim: The requested dimensionality.
        dtype: The dtype of the output array.

    Returns:
        A numpy array of the requested dimensionality and data type.
    """
    assert ndim in (2, 3)
    array = ensure_array(array, dtype)
    if ndim == 2:
        assert array.ndim in (2, 3, 4, 5), str(array.ndim)
        if array.ndim == 3:
            assert array.shape[0] == 1
            array = array[0]
        elif array.ndim == 4:
            assert array.shape[:2] == (1, 1)
            array = array[0, 0]
        elif array.ndim == 5:
            assert array.shape[:3] == (1, 1, 1)
            array = array[0, 0, 0]
    else:
        assert array.ndim in (3, 4, 5), str(array.ndim)
        if array.ndim == 4:
            assert array.shape[0] == 1, f"{array.shape}"
            array = array[0]
        elif array.ndim == 5:
            assert array.shape[:2] == (1, 1)
            array = array[0, 0]
    return array


def ensure_patch_shape(
    raw: np.ndarray,
    labels: Optional[np.ndarray],
    patch_shape: Tuple[int, ...],
    have_raw_channels: bool = False,
    have_label_channels: bool = False,
    channel_first: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Ensure that the raw data and labels have at least the requested patch shape.

    If either raw data or labels do not have the patch shape they will be padded.

    Args:
        raw: The input raw data.
        labels: The input labels.
        patch_shape: The required minimal patch shape.
        have_raw_channels: Whether the raw data has channels.
        have_label_channels: Whether the label data has channels.
        channel_first: Whether the channel axis is the first or last axis.

    Returns:
        The raw data.
        The labels.
    """
    raw_shape = raw.shape
    if labels is not None:
        labels_shape = labels.shape

    # In case the inputs has channels and they are channels first
    # IMPORTANT: for ImageCollectionDataset
    if have_raw_channels and channel_first:
        raw_shape = raw_shape[1:]

    if have_label_channels and channel_first and labels is not None:
        labels_shape = labels_shape[1:]

    # Extract the pad_width and pad the raw inputs
    if any(sh < psh for sh, psh in zip(raw_shape, patch_shape)):
        pw = [(0, max(0, psh - sh)) for sh, psh in zip(raw_shape, patch_shape)]

        if have_raw_channels and channel_first:
            pad_width = [(0, 0), *pw]
        elif have_raw_channels and not channel_first:
            pad_width = [*pw, (0, 0)]
        else:
            pad_width = pw

        raw = np.pad(array=raw, pad_width=pad_width)

    # Extract the pad width and pad the label inputs
    if labels is not None and any(sh < psh for sh, psh in zip(labels_shape, patch_shape)):
        pw = [(0, max(0, psh - sh)) for sh, psh in zip(labels_shape, patch_shape)]

        if have_label_channels and channel_first:
            pad_width = [(0, 0), *pw]
        elif have_label_channels and not channel_first:
            pad_width = [*pw, (0, 0)]
        else:
            pad_width = pw

        labels = np.pad(array=labels, pad_width=pad_width)
    if labels is None:
        return raw
    else:
        return raw, labels


def get_constructor_arguments(obj):
    """@private
    """
    # All relevant torch_em classes have 'init_kwargs' to directly recover the init call.
    if hasattr(obj, "init_kwargs"):
        return getattr(obj, "init_kwargs")

    def _get_args(obj, param_names):
        return {name: getattr(obj, name) for name in param_names}

    # We don't need to find the constructor arguments for optimizers/schedulers because we deserialize the state later.
    if isinstance(
        obj, (
            torch.optim.Optimizer,
            torch.optim.lr_scheduler._LRScheduler,
            # ReduceLROnPlateau does not inherit from _LRScheduler
            torch.optim.lr_scheduler.ReduceLROnPlateau
        )
    ):
        return {}

    # recover the arguments for torch dataloader
    elif isinstance(obj, torch.utils.data.DataLoader):
        # These are all the "simple" arguements.
        # "sampler", "batch_sampler" and "worker_init_fn" are more complicated
        # and generally not used in torch_em
        return _get_args(
            obj, [
                "batch_size", "shuffle", "num_workers", "pin_memory", "drop_last",
                "persistent_workers", "prefetch_factor", "timeout"
            ]
        )

    # TODO support common torch losses (e.g. CrossEntropy, BCE)
    warnings.warn(
        f"Constructor arguments for {type(obj)} cannot be deduced.\n" +
        "For this object, empty constructor arguments will be used.\n" +
        "The trainer can probably not be correctly deserialized via 'DefaultTrainer.from_checkpoint'."
    )
    return {}


def get_trainer(checkpoint: str, name: str = "best", device: Optional[str] = None):
    """Load trainer from a checkpoint.

    Args:
        checkpoint: The path to the checkpoint.
        name: The name of the checkpoint.
        device: The device to use for loading the checkpoint.

    Returns:
        The trainer.
    """
    # try to load from file
    if isinstance(checkpoint, str):
        assert os.path.exists(checkpoint), checkpoint
        trainer = torch_em.trainer.DefaultTrainer.from_checkpoint(checkpoint, name=name, device=device)
    else:
        trainer = checkpoint
    assert isinstance(trainer, torch_em.trainer.DefaultTrainer)
    return trainer


def get_normalizer(trainer):
    """@private
    """
    dataset = trainer.train_loader.dataset
    while (
        isinstance(dataset, torch_em.data.concat_dataset.ConcatDataset) or
        isinstance(dataset, torch.utils.data.dataset.ConcatDataset)
    ):
        dataset = dataset.datasets[0]

    if isinstance(dataset, torch.utils.data.dataset.Subset):
        dataset = dataset.dataset

    preprocessor = dataset.raw_transform

    if hasattr(preprocessor, "normalizer"):
        return preprocessor.normalizer
    else:
        return preprocessor


def load_model(
    checkpoint: str,
    model: Optional[torch.nn.Module] = None,
    name: str = "best",
    state_key: str = "model_state",
    device: Optional[str] = None,
) -> torch.nn.Module:
    """Load model from a trainer checkpoint.

    This function can either load the model directly from the trainer (model is not passed),
    or deserialize the model state from the trainer and load the model state (model is passed).

    Args:
        checkpoint: The path to the checkpoint folder.
        model: The model for which the state should be loaded.
            If it is not passed, the model class and parameters will also be loaded from the trainer.
        name: The name of the checkpoint.
        state_key: The name of the model state to load.
        device: The device on which to load the model.

    Returns:
        The model.
    """
    if model is None:  # load the model and its state from the checkpoint
        model = get_trainer(checkpoint, name=name, device=device).model

    else:  # load the model state from the checkpoint
        if os.path.isdir(checkpoint):
            ckpt = os.path.join(checkpoint, f"{name}.pt")
        else:
            ckpt = checkpoint

        state = torch.load(ckpt, map_location=device)[state_key]

        # to enable loading compiled models
        compiled_prefix = "_orig_mod."
        state = OrderedDict(
            [(k[len(compiled_prefix):] if k.startswith(compiled_prefix) else k, v) for k, v in state.items()]
        )
        model.load_state_dict(state)
        if device is not None:
            model.to(device)
        model.load_state_dict(state)

    return model


def model_is_equal(model1, model2):
    """@private
    """
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def get_random_colors(labels: np.ndarray) -> colors.ListedColormap:
    """Generate a random color map for a label image.

    Args:
        labels: The labels.

    Returns:
        The color map.
    """
    n_labels = len(np.unique(labels)) - 1
    cmap = [[0, 0, 0]] + np.random.rand(n_labels, 3).tolist()
    cmap = colors.ListedColormap(cmap)
    return cmap
