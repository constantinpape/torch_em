import napari
from .util import ensure_array


def check_loader(loader, n_samples, instance_labels=False):
    for ii, (x, y) in enumerate(loader):
        if ii >= n_samples:
            break

        x = ensure_array(x).squeeze(0)
        y = ensure_array(y).squeeze(0)

        v = napari.Viewer()
        v.add_image(x)
        if instance_labels:
            v.add_labels(y.astype("uint32"))
        else:
            v.add_image(y)
        napari.run()
