import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from tiny_eqa.data.common import NumpyTensor


BBox = tuple[int, int, int, int] # TLBR format


def combine_bmasks(bmasks: NumpyTensor['n', 'h', 'w'], sort=False) -> NumpyTensor['h w']:
    """
    """
    cmask = np.zeros_like(bmasks[0], dtype=int)
    if sort:
        bmasks = sorted(bmasks, key=lambda x: x.sum(), reverse=True)
    for i, bmask in enumerate(bmasks):
        cmask[bmask] = i + 1
    return cmask


BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED   = (255,   0,   0)
GREEN = (  0, 255,   0)
BLUE  = (  0,   0, 255)


def visualize_image(image: NumpyTensor['h', 'w', 3]) -> Image.Image:
    """
    """
    return Image.fromarray(image.astype(np.uint8))


def visualize_tiles(tiles: list[NumpyTensor['h', 'w', 3]], r: int, c: int) -> Image.Image:
    """
    """
    h, w = tiles[0].shape[:2]
    assert all(tile.shape[:2] == (h, w) for tile in tiles), 'tiles must have the same shape'
    image = np.concatenate([
        np.concatenate(tiles[cindex * c:(cindex + 1) * c], axis=1)
        for cindex in range(r)
    ], axis=0)
    return Image.fromarray(image.astype(np.uint8))


def visualize_depth(depth: NumpyTensor['h', 'w']) -> Image.Image:
    """
    """
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = np.clip(depth, 0, 1)
    depth = np.stack([depth] * 3, axis=-1)
    return Image.fromarray((depth * 255).astype(np.uint8))


def visualize_cmask(
    cmask: NumpyTensor['h', 'w'], 
    image: NumpyTensor['h', 'w' ,3]=None, background=BLACK, foreground=None, blend=0.25
) -> Image.Image:
    """
    """
    # deterministic background color
    palette = np.random.RandomState(0).randint(0, 255, (np.max(cmask) + 1, 3))
    palette[0] = background
    if foreground is not None:
        for i in range(1, len(palette)):
            palette[i] = foreground
    image_mask = palette[cmask.astype(int)] # type conversion for boolean masks
    image_blend = image_mask if image is None else image_mask * (1 - blend) + image * blend
    image_blend = np.clip(image_blend, 0, 255).astype(np.uint8)
    return Image.fromarray(image_blend)


def visualize_bmask(bmask: NumpyTensor['h', 'w']) -> Image.Image:
    """
    """
    return visualize_cmask(bmask, background=BLACK, foreground=WHITE)


def visualize_bmasks(
    masks: NumpyTensor['n', 'h', 'w'], 
    image: NumpyTensor['h', 'w', 3]=None, background=BLACK, blend=0.25
) -> Image.Image:
    """
    """
    cmask = combine_bmasks(masks, sort=True)
    return visualize_cmask(cmask, image, background=background, blend=blend)


def visualize_bbox(bbox: BBox, image: NumpyTensor['h', 'w', 3], color=GREEN) -> Image.Image:
    """
    """
    image_bbox = image.copy()
    cv2.rectangle(image_bbox, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 2)
    return Image.fromarray(image_bbox)


def visualize_bboxes(bboxes: list[BBox], image: NumpyTensor['h', 'w', 3], color=GREEN) -> Image.Image:
    """
    """
    image_bboxes = image.copy()
    for bbox in bboxes:
        cv2.rectangle(image_bboxes, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 2)
    return Image.fromarray(image_bboxes)


def visualize_point_cloud(points, colors=None, size=1):
    """
    """
    colors = np.full(points.shape, 0.5) if colors is None else colors / 255.0

    trace = go.Scatter3d(
        x=points[:, 0], 
        y=points[:, 2], 
        z=points[:, 1],
        mode='markers',
        marker=dict(
            size=size,
            color=['rgb({},{},{})'.format(r, g, b) for r, g, b in colors * 255],
            opacity=0.8
        )
    )
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            zaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
    )
    fig = go.Figure(data=[trace], layout=layout)
    return fig


def visualize_poses(poses: NumpyTensor['n', 4, 4], size=1):
    """
    """
    origins, directions = poses[:, :3, 3], poses[:, :3, :3] @ np.array([[0, 0, -1]]).T

    trace_directions = go.Cone(
        x=origins[:, 0], 
        y=origins[:, 2],
        z=origins[:, 1],
        u=directions[:, 0].ravel(), 
        v=directions[:, 2].ravel(), 
        w=directions[:, 1].ravel(),
        colorscale='Viridis', sizemode='scaled', sizeref=0.1, showscale=False
    )
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            zaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
    )
    fig = go.Figure(data=[trace_directions], layout=layout)
    return fig


def plot_points(image: Image.Image, points: NumpyTensor['n', 2], color=RED, size=1) -> Image.Image:
    """
    """
    image = np.array(image)
    for point in points:
        cv2.circle(image, tuple(point), size, color, -1)
    return Image.fromarray(image)


def plot_outline(cmask: NumpyTensor['h', 'w'], image: NumpyTensor['h', 'w', 3], color=RED) -> Image.Image:
    """
    """
    cmask = cmask.astype(np.int32)
    outline = np.zeros_like(cmask, dtype=np.uint8)
    class_labels = np.unique(cmask)

    # Define a structuring element for morphological operations
    kernel = np.ones((3, 3), dtype=np.uint8)

    # For each class label, find its edges
    for label in class_labels:
        # Skip background if label is meant for background
        # if label == background_label:
        #     continue

        # Create a binary mask for the current class
        class_mask = (cmask == label).astype(np.uint8)

        # Find edges using morphological gradient
        gradient = cv2.morphologyEx(class_mask, cv2.MORPH_GRADIENT, kernel)

        # Add the gradient to the outline image
        outline = np.maximum(outline, gradient)

    image[outline == 1] = color
    return image