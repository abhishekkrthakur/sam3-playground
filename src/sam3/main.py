from functools import lru_cache
from typing import Dict, Tuple

import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor



def _resolve_device() -> str:
    """Pick the best available device for inference."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def load_model() -> Tuple[Sam3Model, Sam3Processor, str]:
    """Load and cache the SAM3 model and processor."""
    device = _resolve_device()
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    return model, processor, device


def run_inference(
    image: Image.Image,
    prompt: str,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    boxes=None,
    box_labels=None,
) -> Dict:
    """Run SAM3 instance segmentation for a prompt and thresholds."""
    threshold = float(max(0.0, min(threshold, 1.0)))
    mask_threshold = float(max(0.0, min(mask_threshold, 1.0)))

    model, processor, device = load_model()
    input_kwargs = {"images": image, "return_tensors": "pt"}

    if prompt:
        input_kwargs["text"] = prompt
    if boxes:
        input_kwargs["input_boxes"] = [boxes]
        labels = box_labels or [1] * len(boxes)
        # Ensure labels length matches boxes length.
        if len(labels) < len(boxes):
            labels = labels + [1] * (len(boxes) - len(labels))
        input_kwargs["input_boxes_labels"] = [labels[: len(boxes)]]

    inputs = processor(**input_kwargs).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    return result
