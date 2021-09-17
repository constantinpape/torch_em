import os

import numpy as np
import torch

from elf.segmentation.embeddings import embedding_pca
from skimage.segmentation import mark_boundaries
from torchvision.utils import make_grid

# tensorboard import only works if tensobard package is available, so
# we wrap this in a try except
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from ..util import ensure_tensor
from ..loss import EMBEDDING_LOSSES


def normalize_im(im):
    im = ensure_tensor(im, dtype=torch.float32)
    im -= im.min()
    im /= im.max()
    return im


def make_grid_image(image, y, prediction, selection, gradients=None):
    target_image = normalize_im(y[selection].cpu())
    pred_image = normalize_im(prediction[selection].detach().cpu())

    # TODO support rgb images?
    if image.shape[0] > 1:
        image = image[0:1]

    n_channels = target_image.shape[0]
    if n_channels == 1:
        nrow = 8
        images = [image, target_image, pred_image]
    else:
        nrow = n_channels
        images = nrow * [image]
        images += [channel.unsqueeze(0) for channel in target_image]
        images += [channel.unsqueeze(0) for channel in pred_image]

    if gradients is not None:
        grad_image = normalize_im(gradients[selection].cpu())
        if n_channels == 1:
            images.append(grad_image)
        else:
            images += [channel.unsqueeze(0) for channel in grad_image]

    im = make_grid(images, nrow=nrow, padding=4)
    name = 'raw_targets_predictions'
    if gradients is not None:
        name += '_gradients'
    return im, name


def make_embedding_image(image, y, prediction, selection, gradients=None):
    assert gradients is None, "Not implemented"
    image = image.numpy()

    seg = y[selection].cpu().numpy()
    seg = mark_boundaries(image[0], seg[0])  # need to get rid of singleton channel
    seg = seg.transpose((2, 0, 1))  # to channel first

    pred = prediction[selection].detach().cpu().numpy()
    pca = embedding_pca(pred)

    image = np.repeat(image, 3, axis=0)  # to rgb

    images = [
        torch.from_numpy(im) for im in (image, seg, pca)
    ]
    im = make_grid(images, padding=4)
    name = 'raw_segmentation_embedding'
    if gradients is not None:
        name += '_gradients'
    return im, name


class TensorboardLogger:
    def __init__(self, trainer):
        self.log_dir = f'./logs/{trainer.name}'
        os.makedirs(self.log_dir, exist_ok=True)

        if SummaryWriter is None:
            msg = "Need tensorboard package to use logger. Install it via 'conda install -c conda-forge tensorboard'"
            raise RuntimeError(msg)
        self.tb = torch.utils.tensorboard.SummaryWriter(self.log_dir)
        self.log_image_interval = trainer.log_image_interval

        # derive which visualisation method is appropriate, based on the loss function
        if type(trainer.loss) in EMBEDDING_LOSSES:
            self.have_embeddings = True
            self.make_image = make_embedding_image
        else:
            self.have_embeddings = False
            self.make_image = make_grid_image

    def log_images(self, step, x, y, prediction, name, gradients=None):

        selection = np.s_[0] if x.ndim == 4 else np.s_[0, :, x.shape[2] // 2]

        image = normalize_im(x[selection].cpu())
        self.tb.add_image(tag=f'{name}/input',
                          img_tensor=image,
                          global_step=step)

        im, im_name = self.make_image(image, y, prediction, selection, gradients)
        im_name = f'{name}/{im_name}'
        self.tb.add_image(tag=im_name, img_tensor=im, global_step=step)

    def log_train(self, step, loss, lr, x, y, prediction, log_gradients=False):
        self.tb.add_scalar(tag='train/loss', scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag='train/learning_rate', scalar_value=lr, global_step=step)

        # the embedding visualisation function currently doesn't support gradients,
        # so we can't log them even if log_gradients is true
        log_grads = log_gradients and self.have_embeddings
        if step % self.log_image_interval == 0:
            gradients = prediction.grad if log_grads else None
            self.log_images(step, x, y, prediction, 'train', gradients=gradients)

    def log_validation(self, step, metric, loss, x, y, prediction):
        self.tb.add_scalar(tag='validation/loss', scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag='validation/metric', scalar_value=metric, global_step=step)
        self.log_images(step, x, y, prediction, 'validation')
