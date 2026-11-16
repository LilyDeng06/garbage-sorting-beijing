"""Single-image inference using a trusted checkpoint from train_garbage265.py."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torchvision import models
from PIL import Image, ImageOps


def predict(checkpoint, image_path, device="cpu", topk=5):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = ck["classes"]
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    transform = models.EfficientNet_B0_Weights.DEFAULT.transforms()
    with Image.open(image_path) as im:
        x = transform(ImageOps.exif_transpose(im).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores, indices = model(x).softmax(1)[0].topk(min(topk, len(classes)))
    candidates = []
    for score, index in zip(scores.tolist(), indices.tolist()):
        label = classes[index]
        coarse, _, item = label.partition("-")
        candidates.append(dict(class_id=index, label=label, item=item,
                               waste_category=coarse.replace("其它垃圾", "其他垃圾"), score=score))
    return dict(image=str(image_path), prediction=candidates[0], topk=candidates,
                note="Scores are uncalibrated softmax outputs; waste_category follows dataset labels, not local disposal rules.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--topk", type=int, default=5)
    a = p.parse_args()
    if a.topk < 1:
        p.error("--topk must be positive")
    print(json.dumps(predict(a.checkpoint, a.image, a.device, a.topk), ensure_ascii=False, indent=2))
