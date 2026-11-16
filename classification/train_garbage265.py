"""CSV-based EfficientNet-B0 transfer learning; validation is optional.
Requires PyTorch >=2.2, matching torchvision, Pillow and tqdm.
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageOps
from tqdm import tqdm


class WasteDataset(Dataset):
    def __init__(self, repo, cache, split, classes, transform):
        self.transform = transform
        self.samples = []
        roots = sorted(p.parent for p in cache.glob(f"**/extracted/*/{split}") if p.is_dir())
        if not roots:
            raise RuntimeError(f"No extracted {split} directory under {cache}")
        missing = []
        with (repo / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                relative = Path(row["image:FILE"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Invalid image path: {relative}")
                label = int(row["category"])
                if not 0 <= label < len(classes):
                    raise ValueError(f"Invalid category: {label}")
                path = next((r / relative for r in roots if (r / relative).is_file() and (r / relative).stat().st_size), None)
                if path is None:
                    missing.append(str(relative))
                else:
                    self.samples.append((path, label))
        if missing:
            raise RuntimeError(f"{split}: {len(missing)} missing/empty images; wait for extraction. Examples: {missing[:3]}")
        if not self.samples:
            raise RuntimeError(f"Empty {split} dataset")
        print(f"{split}: {len(self.samples)} files found", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                return self.transform(image), label
        except Exception as exc:
            raise RuntimeError(f"Cannot decode/transform image: {path}") from exc


def atomic_save(value, path):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.home()/"datasets/garbage265")
    p.add_argument("--cache", type=Path, default=Path.home()/"datasets/garbage265_cache")
    p.add_argument("--output", type=Path, default=Path.home()/"garbage_runs/efficientnet_b0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--head-epochs", type=int, default=3)
    p.add_argument("--finetune-epochs", type=int, default=12)
    p.add_argument("--validate", action="store_true", help="Requires fully extracted val; default: train only")
    p.add_argument("--resume", type=Path, help="Trusted last.pt from this script; resumes at next epoch")
    p.add_argument("--weights", type=Path, help="Local official ImageNet state_dict; otherwise automatic download")
    p.add_argument("--check-only", action="store_true", help="Verify all selected images decode; no GPU/weights needed")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.batch_size < 2 or a.workers < 0 or a.head_epochs < 0 or a.finetune_epochs < 0 or a.head_epochs+a.finetune_epochs < 1:
        p.error("Invalid batch/workers/epochs")
    torch.set_num_threads(4)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    classes = (a.repo/"classname.txt").read_text(encoding="utf-8-sig").splitlines()
    classes = [s.strip() for s in classes]
    if len(classes) != 265 or any(not s for s in classes):
        raise ValueError("classname.txt must contain exactly 265 nonempty lines in label-ID order")
    coarse_names = [c.split("-", 1)[0].replace("其它垃圾", "其他垃圾") for c in classes]
    bins = sorted(set(coarse_names))
    if len(bins) != 4:
        raise ValueError(f"Expected four class-name prefixes, found {bins}")
    bin_ids = torch.tensor([bins.index(name) for name in coarse_names])
    weights = models.EfficientNet_B0_Weights.DEFAULT
    tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406], [.229,.224,.225]),
    ])
    train = WasteDataset(a.repo, a.cache, "train", classes, tf)
    val = WasteDataset(a.repo, a.cache, "val", classes, weights.transforms()) if a.validate else None
    if a.check_only:
        for dataset in [train, val]:
            if dataset is None:
                continue
            for path, _ in tqdm(dataset.samples, desc="Checking image decoding"):
                with Image.open(path) as im:
                    ImageOps.exif_transpose(im).convert("RGB").load()
        print("All selected images decoded successfully.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Activate the lab GPU PyTorch environment first.")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}; validation={a.validate}", flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    if not a.resume and (a.output/"last.pt").exists():
        raise RuntimeError("Output already contains last.pt. Use --resume or a new --output.")
    ck = torch.load(a.resume, map_location="cpu", weights_only=False) if a.resume else None
    if ck and (ck["classes"] != classes or ck["head_epochs"] != a.head_epochs):
        raise ValueError("Resume class order/head-epochs differs from checkpoint")
    model = models.efficientnet_b0(weights=None if ck or a.weights else weights)
    if a.weights and not ck:
        model.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True))
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 265)
    if ck:
        model.load_state_dict(ck["model"])
    model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    if ck:
        optimizer.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["rng"])
        # This trainer uses only logical cuda:0. Older checkpoints may contain
        # RNG states for all four visible GPUs; resume can expose only one.
        if ck.get("cuda_rng"):
            torch.cuda.set_rng_state(ck["cuda_rng"][0], device=device)
        random.setstate(ck["python_rng"])
    kwargs = dict(batch_size=a.batch_size, num_workers=a.workers, pin_memory=True)
    loader = DataLoader(train, shuffle=True, **kwargs)
    vloader = DataLoader(val, shuffle=False, **kwargs) if val is not None else None
    criterion = nn.CrossEntropyLoss()
    bin_ids = bin_ids.to(device)
    start = ck["epoch"]+1 if ck else 0
    total_epochs = a.head_epochs+a.finetune_epochs
    for epoch in range(start, total_epochs):
        frozen = epoch < a.head_epochs
        for param in model.features.parameters():
            param.requires_grad_(not frozen)
        optimizer.param_groups[1]["lr"] = 1e-3 if frozen else 1e-4
        model.train()
        if frozen:
            model.features.eval()  # Freeze BatchNorm statistics and stochastic depth too.
        loss_sum = correct = count = 0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; last completed epoch remains in last.pt")
            loss.backward()
            optimizer.step()
            count += labels.numel()
            correct += (logits.argmax(1)==labels).sum().item()
            loss_sum += loss.item()*labels.numel()
        metrics = dict(epoch=epoch+1, stage="head" if frozen else "finetune", train_loss=loss_sum/count, train_accuracy=correct/count)
        if vloader is not None:
            model.eval()
            n = fine = coarse = 0
            with torch.inference_mode():
                for images, labels in tqdm(vloader, desc="Validation"):
                    labels = labels.to(device)
                    pred = model(images.to(device)).argmax(1)
                    n += labels.numel()
                    fine += (pred==labels).sum().item()
                    coarse += (bin_ids[pred]==bin_ids[labels]).sum().item()
            metrics.update(val_accuracy=fine/n, val_bin_accuracy=coarse/n)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                     classes=classes, bins=bins, head_epochs=a.head_epochs, architecture="efficientnet_b0",
                     image_size=224, rng=torch.get_rng_state(), cuda_rng=[torch.cuda.get_rng_state(device)],
                     python_rng=random.getstate(), metrics=metrics,
                     args={k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()})
        atomic_save(state, a.output/"last.pt")
        atomic_save({"model": state["model"], "classes": classes, "bins": bins,
                     "architecture": "efficientnet_b0", "epoch": epoch, "metrics": metrics},
                    a.output/f"epoch_{epoch+1:03d}.pt")
        with (a.output/"history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics)+"\n")
        print(json.dumps(metrics), flush=True)
    print("Training finished. Select a checkpoint using validation, not training accuracy.", flush=True)


if __name__ == "__main__":
    main()
