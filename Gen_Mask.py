import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

feature_extractor = SegformerFeatureExtractor.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
)
model.eval()

CUSTOM_COLORS = {
    0: (0, 0, 0),  # unlabeled
    1: (255, 0, 180),  # road → pink
    2: (255, 0, 180),  # sidewalk (optional)
    3: (128, 128, 128),  # building → gray
    8: (0, 128, 255),  # sky → blue
    10: (0, 255, 0),  # vegetation/trees → green
    13: (255, 0, 0),  # car → red
}


def segment_cityscapes(img_path):
    image = Image.open(img_path).convert("RGB")

    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False
    )

    seg = upsampled.argmax(dim=1)[0].cpu().numpy()

    # Create colored mask
    h, w = seg.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id in np.unique(seg):
        color = CUSTOM_COLORS.get(cls_id, (0, 0, 0))
        mask[seg == cls_id] = color

    return image, mask


def Gen_Mask(Image):
    Full_mask = []
    for i in range(len(Image)):
        Img = Image[i]
        orig, mask = segment_cityscapes(Img)
        Full_mask.append(mask)

    return np.asarray(Full_mask)
