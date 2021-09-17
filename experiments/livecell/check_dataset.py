from torch_em.data.datasets.livecell import get_livecell_loader
from torch_em.util.debug import check_loader

PATH = "/home/pape/Work/data/livecell"


def check_livecell_size():
    patch_shape = (512, 512)
    loader = get_livecell_loader(PATH, patch_shape, "train", download=True, batch_size=1)
    print("Training images:", len(loader.dataset))

    loader = get_livecell_loader(PATH, patch_shape, "val", download=True, batch_size=1)
    print("Val images:", len(loader.dataset))


def check_livecell_images():
    patch_shape = (512, 512)
    loader = get_livecell_loader(PATH, patch_shape, "train", download=True, batch_size=1)
    check_loader(loader, 10, instance_labels=True)


# NOTE:
# - Tischi had a problem containing similar data
# - overlapping instances! are not reflected in the current label processing!
# - there seem to be quite a lot of cells not captured in the segmentation labels
if __name__ == "__main__":
    check_livecell_size()
    check_livecell_images()
