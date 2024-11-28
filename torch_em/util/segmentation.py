import numpy as np

import vigra
import elf.segmentation as elseg
from elf.segmentation.utils import normalize_input
from elf.segmentation.mutex_watershed import mutex_watershed

from skimage.measure import label
from skimage.filters import gaussian
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt


#
# segmentation functionality
#


# could also refactor this into elf
def size_filter(seg, min_size, hmap=None, with_background=False):
    if min_size == 0:
        return seg

    if hmap is None:
        ids, sizes = np.unique(seg, return_counts=True)
        bg_ids = ids[sizes < min_size]
        seg[np.isin(seg, bg_ids)] = 0
        seg, _, _ = vigra.analysis.relabelConsecutive(seg.astype(np.uint), start_label=1, keep_zeros=True)
    else:
        assert hmap.ndim in (seg.ndim, seg.ndim + 1)
        hmap_ = np.max(hmap[:seg.ndim], axis=0) if hmap.ndim > seg.ndim else hmap
        if with_background:
            seg, _ = elseg.watershed.apply_size_filter(seg + 1, hmap_, min_size, exclude=[1])
            seg[seg == 1] = 0
        else:
            seg, _ = elseg.watershed.apply_size_filter(seg, hmap_, min_size)
    return seg


def mutex_watershed_segmentation(foreground, affinities, offsets, min_size=50, threshold=0.5, strides=None):
    """Computes the mutex watershed segmentation using the affinity maps for respective pixel offsets

    Arguments:
        - foreground: [np.ndarray] - The foreground background channel for the objects
        - affinities [np.ndarray] - The input affinity maps
        - offsets: [list[list[int]]] - The pixel offsets corresponding to the affinity channels
        - min_size: [int] - The minimum pixels (below which) to filter objects
        - threshold: [float] - To threshold foreground predictions
    """
    mask = (foreground >= threshold)
    if strides is None:
        strides = [2] * foreground.ndim
    seg = mutex_watershed(affinities, offsets=offsets, mask=mask, strides=strides, randomize_strides=True)
    seg = size_filter(seg.astype("uint32"), min_size=min_size, hmap=affinities, with_background=True)
    return seg


def multicut_watershed_segmentation(boundaries):
    """Compute the multicut watershed segmentation using the boundary maps.
    """
    ws_seg, _ = elseg.watershed.distance_transform_watershed(input_=boundaries, threshold=0.25, sigma_seeds=2.0)

    # Compute the region adjacency graph
    rag = elseg.features.compute_rag(ws_seg)

    # Compute the edge costs
    costs = elseg.features.compute_boundary_features(rag, boundaries)[:, 0]

    # Transform the edge costs from [0, 1] to [-inf, inf], which is necessaru for multicut.
    # This is done by interpreting the values as probabilities for an edge being 'true'
    # and then taking the negative log-likelihood. In addition, we weight the costs by
    # the size of the corresponding edge for z and xy edges.
    z_edges = elseg.features.compute_z_edge_mask(rag, ws_seg)
    xy_edges = np.logical_not(z_edges)
    edge_population = [z_edges, xy_edges]

    edge_sizes = elseg.features.compute_boundary_mean_and_length(rag, boundaries)[:, 1]

    costs = elseg.multicut.transform_probabilities_to_costs(
        costs, edge_sizes=edge_sizes, edge_populations=edge_population,
    )

    # Run the multicut partitioning
    # Here, we use the kernighan lin heuristics to solve the problem, introduced in
    # http://xilinx.asia/_hdl/4/eda.ee.ucla.edu/EE201A-04Spring/kl.pdf
    node_labels = elseg.multicut.multicut_kernighan_lin(rag, costs)

    # Map the results back to pixels to obtain the final segmentation
    instances = elseg.features.project_node_labels_to_pixels(rag, node_labels)
    return instances


def connected_components_with_boundaries(foreground, boundaries, threshold=0.5):
    input_ = np.clip(foreground - boundaries, 0, 1)
    seeds = label(input_ > threshold)
    mask = normalize_input(foreground > threshold)
    seg = watershed(boundaries, markers=seeds, mask=mask)
    return seg.astype("uint64")


def watershed_from_components(boundaries, foreground, min_size=50, threshold1=0.5, threshold2=0.5):
    """The default approach:
    - Subtract the boundaries from the foreground to separate touching objects.
    - Use the connected components of this as seeds.
    - Use the thresholded foreground predictions as mask to grow back the pieces
      lost by subtracting the boundary prediction.

    Arguments:
        - boundaries: [np.ndarray] - The boundaries for objects
        - foreground: [np.ndarray] - The foregrounds for objects
        - min_size: [int] - The minimum pixels (below which) to filter objects
        - threshold1: [float] - To separate touching objects (by subtracting bd and fg) above threshold
        - threshold2: [float] - To threshold foreground predictions

    Returns:
        seg: [np.ndarray] - instance segmentation
    """
    seeds = label((foreground - boundaries) > threshold1)
    mask = foreground > threshold2
    seg = watershed(boundaries, seeds, mask=mask)
    seg = size_filter(seg, min_size)
    return seg


def watershed_from_maxima(boundaries, foreground, min_distance, min_size=50, sigma=1.0, threshold1=0.5):
    """Find objects via seeded watershed starting from the maxima of the distance transform instead.
    This has the advantage that objects can be better separated, but it may over-segment
    if the objects have complex shapes.

    The min_distance parameter controls the minimal distance between seeds, which
    corresponds to the minimal distance between object centers.

    Arguments:
        - boundaries: [np.ndarray] - The boundaries for objects
        - foreground: [np.ndarray] - The foreground for objects
        - min_size: [int] - min. pixels (below which) to filter objects
        - min_distance: [int] - min. distance of peaks (see `from skimage.feature import peak_local_max`)
        - sigma: [float] - standard deviation for gaussian kernel. (see `from skimage.filters import gaussian`)
        - threshold1: [float] - To threshold foreground predictions

    Returns
        seg: [np.ndarray] - instance segmentation
    """
    mask = foreground > threshold1
    boundary_distances = distance_transform_edt(boundaries < 0.1)
    boundary_distances[~mask] = 0  # type: ignore
    boundary_distances = gaussian(boundary_distances, sigma)  # type: ignore
    seed_points = peak_local_max(boundary_distances, min_distance=min_distance, exclude_border=False)
    seeds = np.zeros(mask.shape, dtype="uint32")
    seeds[seed_points[:, 0], seed_points[:, 1]] = np.arange(1, len(seed_points) + 1)
    seg = watershed(boundaries, markers=seeds, mask=foreground)
    seg = size_filter(seg, min_size)
    return seg


def watershed_from_center_and_boundary_distances(
    center_distances,
    boundary_distances,
    foreground_map,
    center_distance_threshold=0.5,
    boundary_distance_threshold=0.5,
    foreground_threshold=0.5,
    distance_smoothing=1.6,
    min_size=0,
    debug=False,
):
    """Seeded watershed based on distance predictions to object center and boundaries.

    The seeds are computed by finding connected components where both distance predictions
    are smaller than the respective thresholds. Using both distances here should prevent merging
    narrow adjacent objects (if only using the center distance) or finding multiple seeds for non-convex
    cells (if only using the boundary distances).

    Args:
        center_distances [np.ndarray] - Distance prediction to the objcet center.
        boundary_distances [np.ndarray] - Inverted distance prediction to object boundaries.
        foreground_map [np.ndarray] - Predictio for foreground probabilities.
        center_distance_threshold [float] - Center distance predictions below this value will be
            used to find seeds (intersected with thresholded boundary distance predictions).
        boundary_distance_threshold [float] - Boundary distance predictions below this value will be
            used to find seeds (intersected with thresholded center distance predictions).
        foreground_threshold [float] - Foreground predictions above this value will be used as foreground mask.
        distance_smoothing [float] - Sigma value for smoothing the distance predictions.
        min_size [int] - Minimal object size in the segmentation result.
        debug [bool] - Return all intermediate results for debugging.

    Returns:
        np.ndarray - The instance segmentation.
    """
    center_distances = vigra.filters.gaussianSmoothing(center_distances, distance_smoothing)
    boundary_distances = vigra.filters.gaussianSmoothing(boundary_distances, distance_smoothing)

    fg_mask = foreground_map > foreground_threshold

    marker_map = np.logical_and(
        center_distances < center_distance_threshold,
        boundary_distances < boundary_distance_threshold
    )
    marker_map[~fg_mask] = 0
    markers = label(marker_map)

    seg = watershed(boundary_distances, markers=markers, mask=fg_mask)
    seg = size_filter(seg, min_size)

    if debug:
        debug_output = {
            "center_distances": center_distances,
            "boundary_distances": boundary_distances,
            "foreground_mask": fg_mask,
            "markers": markers,
        }
        return seg, debug_output

    return seg
