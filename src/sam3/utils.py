import itertools

import numpy as np
import matplotlib
from PIL import Image, ImageDraw


def _generate_colors(n_masks: int):
    """Generate deterministic colors so masks and boxes share a palette."""
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(max(1, n_masks))
    return [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]


def overlay_masks(image, masks, colors=None):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)

    colors = colors or _generate_colors(masks.shape[0])
    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image


def overlay_boxes(image, masks, boxes, colors=None):
    """Draw bounding boxes that match the mask color palette."""
    img = image.convert("RGBA")
    draw = ImageDraw.Draw(img)
    colors = colors or _generate_colors(masks.shape[0])

    def _to_xyxy(box, width, height):
        """Normalize incoming boxes to xyxy pixel coords."""
        # Flatten [[x1,y1], [x2,y2], ...] into a flat list.
        if box and isinstance(box[0], (list, tuple)):
            flat = list(itertools.chain.from_iterable(box))
        else:
            flat = list(box)

        # We only need the first four numbers; extra values get ignored.
        flat = [float(v) for v in flat[:4]] + [0.0] * max(0, 4 - len(flat))
        x0, y0, x1, y1 = flat[:4]

        # If everything looks normalized (<= 1), scale to pixel space.
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
            x0, y0, x1, y1 = x0 * width, y0 * height, x1 * width, y1 * height

        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))

        # Clamp to image bounds to avoid "off-canvas" drawing.
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        return [x0, y0, x1, y1]

    width, height = image.size
    for color, box in zip(colors, boxes):
        xyxy = _to_xyxy(box, width, height)
        draw.rectangle(xyxy, outline=color + (255,), width=3)

    return img
