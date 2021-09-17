import sys
import time
from copy import deepcopy
from shutil import copyfile

import torch
import torch_em

from torch_em.data.datasets import get_covid_if_loader
from torch_em.trainer import DefaultTrainer
from multi_tensor_unet import MultiTensorUNet

DATA_FOLDER = "./data"


class MultiTensorTrainer(DefaultTrainer):
    def _compute_loss(self, p1, p2, y1, y2, loss_function):
        l1 = loss_function(p1, y1)
        l2 = loss_function(p2, y2)
        loss = (l1 + l2) / 2
        return loss

    def _train_epoch(self, progress):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            extra_raw, extra_y = next(iter(self.extra_train_loader))
            extra_raw, extra_y = extra_raw.to(self.device), extra_y.to(self.device)

            self.optimizer.zero_grad()
            p1, p2 = self.model([x, extra_raw])

            loss = self._compute_loss(p1, p2, y, extra_y, self.loss)
            loss.backward()
            self.optimizer.step()

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _validate(self):
        self.model.eval()

        metric = 0.
        loss = 0.

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                extra_raw, extra_y = next(iter(self.extra_train_loader))
                extra_raw, extra_y = extra_raw.to(self.device), extra_y.to(self.device)
                p1, p2 = self.model([x, extra_raw])

                loss += self._compute_loss(p1, p2, y, extra_y, self.loss).item()
                metric += self._compute_loss(p1, p2, y, extra_y, self.metric).item()

        metric /= len(self.val_loader)
        loss /= len(self.val_loader)
        return metric


def train_model():
    patch_shape = (512, 512)
    batch_size = 4
    cell_loader = get_covid_if_loader(DATA_FOLDER, patch_shape=patch_shape, target="cells",
                                      shuffle=False, batch_size=batch_size,
                                      binary=True)
    nuc_loader = get_covid_if_loader(DATA_FOLDER, patch_shape=patch_shape, target="nuclei",
                                     shuffle=False, batch_size=batch_size,
                                     binary=True)

    model = MultiTensorUNet(in_channels=2, out_channels=2, depth=3, initial_features=16)
    loss = torch_em.loss.DiceLoss()
    optimizer = torch.optim.Adam(model.parameters())
    metric = torch_em.loss.DiceLoss()

    trainer = MultiTensorTrainer("multi-tensor", cell_loader, cell_loader, model, loss, optimizer, metric,
                                 mixed_precision=False, device=torch.device("cuda"))
    trainer.extra_train_loader = nuc_loader
    trainer.extra_val_loader = nuc_loader

    iterations = 5000
    trainer.fit(iterations)


def export_model():
    import imageio
    import h5py
    from torch_em.util import add_weight_formats, export_biomageio_model, get_default_citations
    from bioimageio.spec.shared import yaml

    with h5py.File("./data/gt_image_000.h5", "r") as f:
        input_data = [
            f["raw/serum_IgG/s0"][:256, :256],
            f["raw/nuclei/s0"][:256, :256],
        ]
    imageio.imwrite("./cover.jpg", input_data[0])

    doc = "Example Model: Different Output Shape"
    cite = get_default_citations(model="UNet2d")

    export_biomageio_model(
        "./checkpoints/multi-tensor",
        "./exported",
        input_data=input_data,
        authors=[{"name": "Constantin Pape; @constantinpape"}],
        tags=["segmentation"],
        license="CC-BY-4.0",
        documentation=doc,
        git_repo="https://github.com/constantinpape/torch-em.git",
        cite=cite,
        covers=["./cover.jpg"],
        input_optional_parameters=False
    )
    add_weight_formats("./exported", ["onnx", "torchscript"])

    rdf_path = "./exported/rdf.yaml"
    with open(rdf_path, "r") as f:
        rdf = yaml.load(f)

    # update the inputs / output descriptions
    rdf["inputs"][0]["name"] = "input0"
    rdf["inputs"][0]["shape"] = {"min": [1, 1, 32, 32], "step": [0, 0, 16, 16]}

    input1 = deepcopy(rdf["inputs"][0])
    input1["name"] = "input1"
    rdf["inputs"].append(input1)

    rdf["outputs"][0]["name"] = "output0"
    rdf["outputs"][0]["shape"] = {"reference_input": "input0", "offset": [0, 0, 0, 0], "scale": [1, 1, 1, 1]}

    output1 = deepcopy(rdf["outputs"][0])
    output1["name"] = "output1"
    output1["shape"]["reference_input"] = "input1"
    rdf["outputs"].append(output1)

    # update the network description
    rdf["source"] = "./multi_tensor_unet.py:MultiTensorUNet"
    rdf["kwargs"] = dict(in_channels=2, out_channels=2, depth=3, initial_features=16)
    copyfile("./multi_tensor_unet.py", "./exported/multi_tensor_unet.py")

    with open(rdf_path, "w") as f:
        yaml.dump(rdf, f)


if __name__ == "__main__":
    train = bool(int(sys.argv[1]))
    if train:
        train_model()
    else:
        export_model()
