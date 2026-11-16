"""Class-agnostic Faster R-CNN, then a 4-class crop classifier.

Train split is the original 2235 images. Validation stays the original 508.
TACO is not included: its 4-bin labels are only an approximate mapping.
"""

import random
import socket
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ROOT = Path("/root/garbage_yolo")
DATA = ROOT / "dataset"
RUN = ROOT / "runs" / "rcnn"
WEIGHTS = ROOT / "weights"
DEVICE = torch.device("cuda:0")
NAMES = ["可回收物", "有害垃圾", "厨余垃圾", "其他垃圾"]
DET_EPOCHS = 12
CLS_EPOCHS = 16
CLS_PATIENCE = 5
IOU_MATCH = 0.5


def log(msg):
    print(msg, flush=True)


def download(url, dest: Path, min_bytes):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        log(f"have {dest.name} {dest.stat().st_size}")
        return
    log(f"GET {url}")
    tmp = dest.with_suffix(".part")
    last = {"n": 0}

    def hook(block, block_size, total):
        got = block * block_size
        if got - last["n"] >= 20_000_000:
            last["n"] = got
            log(f"  {got / 1e6:.0f} MB")

    socket.setdefaulttimeout(120)
    urllib.request.urlretrieve(url, tmp, hook)
    size = tmp.stat().st_size
    if size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download too small: {size}")
    tmp.replace(dest)
    log(f"saved {dest.name} {size}")


def read_yolo(path: Path, width, height):
    boxes, classes = [], []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls < 0 or cls > 3:
            continue
        xc, yc, bw, bh = (float(v) for v in parts[1:5])
        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height
        x1 = min(max(x1, 0.0), width)
        y1 = min(max(y1, 0.0), height)
        x2 = min(max(x2, 0.0), width)
        y2 = min(max(y2, 0.0), height)
        if x2 - x1 >= 1.0 and y2 - y1 >= 1.0:
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)
    return boxes, classes


def pairs(split):
    img_dir = DATA / "images" / split
    lab_dir = DATA / "labels" / split
    images = {
        p.stem: p
        for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    }
    labels = {p.stem: p for p in lab_dir.iterdir() if p.suffix.lower() == ".txt"}
    keys = sorted(set(images) & set(labels))
    return [(images[k], labels[k]) for k in keys]


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-6, None)


def match_flags(pred_boxes, pred_scores, gt_boxes, iou_thr):
    flags = np.zeros(len(pred_boxes), dtype=np.float64)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return flags
    ious = box_iou(pred_boxes, gt_boxes)
    taken = set()
    for pi in np.argsort(-pred_scores):
        row = ious[pi].copy()
        for g in taken:
            row[g] = -1.0
        g = int(row.argmax())
        if row[g] >= iou_thr:
            flags[pi] = 1.0
            taken.add(g)
    return flags


def average_precision(flags, scores, n_gt):
    if n_gt <= 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = np.cumsum(flags[order])
    fp = np.cumsum(1.0 - flags[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        keep = precision[recall >= t]
        ap += float(keep.max()) if keep.size else 0.0
    return ap / 101.0


def prf_at(records, thr, iou_thr):
    tp = fp = fn = 0
    for boxes, scores, gt, _cls in records:
        keep = scores >= thr
        flags = match_flags(boxes[keep], scores[keep], gt, iou_thr)
        tp += int(flags.sum())
        fp += int(len(flags) - flags.sum())
        fn += int(len(gt) - flags.sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1, tp, fp, fn


class DetSet(Dataset):
    def __init__(self, split, train):
        self.items = pairs(split)
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        ip, lp = self.items[index]
        image = Image.open(ip).convert("RGB")
        width, height = image.size
        boxes, classes = read_yolo(lp, width, height)
        tensor = transforms.functional.to_tensor(image)
        if boxes:
            box_t = torch.tensor(boxes, dtype=torch.float32)
            cls_t = torch.tensor(classes, dtype=torch.int64)
        else:
            box_t = torch.zeros((0, 4), dtype=torch.float32)
            cls_t = torch.zeros((0,), dtype=torch.int64)
        if self.train and random.random() < 0.5 and tensor.shape[-1] > 1:
            tensor = torch.flip(tensor, dims=[-1])
            if len(box_t):
                x1 = box_t[:, 0].clone()
                x2 = box_t[:, 2].clone()
                box_t[:, 0] = tensor.shape[-1] - x2
                box_t[:, 2] = tensor.shape[-1] - x1
        target = {
            "boxes": box_t,
            "labels": torch.ones(len(box_t), dtype=torch.int64),
            "cls": cls_t,
        }
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))


def build_detector(coco_path=None):
    # COCO head has 91 classes. Load those weights before swapping in a 2-class head,
    # otherwise the predictor shapes do not match.
    model = fasterrcnn_resnet50_fpn(
        pretrained=False, pretrained_backbone=False, num_classes=91
    )
    if coco_path is not None:
        model.load_state_dict(torch.load(coco_path, map_location="cpu"))
        log("coco weights loaded")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.transform.min_size = (640,)
    model.transform.max_size = 640
    model.roi_heads.score_thresh = 0.01
    model.roi_heads.detections_per_img = 100
    return model


def move_targets(targets, keep_cls):
    moved = []
    classes = []
    for target in targets:
        classes.append(target["cls"])
        item = {
            "boxes": target["boxes"].to(DEVICE),
            "labels": target["labels"].to(DEVICE),
        }
        if keep_cls:
            item["cls"] = target["cls"]
        moved.append(item)
    return moved, classes


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    records = []
    for images, targets in loader:
        images = [image.to(DEVICE) for image in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            records.append(
                (
                    output["boxes"].detach().cpu().numpy(),
                    output["scores"].detach().cpu().numpy(),
                    target["boxes"].numpy(),
                    target["cls"].numpy(),
                )
            )
    n_gt = sum(len(gt) for _b, _s, gt, _c in records)
    flag_chunks, score_chunks = [], []
    for boxes, scores, gt, _cls in records:
        flag_chunks.append(match_flags(boxes, scores, gt, 0.5))
        score_chunks.append(scores)
    flags = np.concatenate(flag_chunks) if flag_chunks else np.zeros(0)
    scores = np.concatenate(score_chunks) if score_chunks else np.zeros(0)
    ap50 = average_precision(flags, scores, n_gt)

    aps = []
    for thr in np.arange(0.5, 1.0, 0.05):
        chunks = [match_flags(b, s, g, float(thr)) for b, s, g, _c in records]
        aps.append(average_precision(np.concatenate(chunks), scores, n_gt))
    ap5095 = float(np.mean(aps)) if aps else 0.0

    best = None
    for thr in np.linspace(0.05, 0.95, 19):
        precision, recall, f1, tp, fp, fn = prf_at(records, float(thr), IOU_MATCH)
        if best is None or f1 > best[0]:
            best = (f1, precision, recall, float(thr), tp, fp, fn)
    model.train()
    return {
        "ap50": float(ap50),
        "ap5095": ap5095,
        "f1": best[0],
        "precision": best[1],
        "recall": best[2],
        "thr": best[3],
        "tp": best[4],
        "fp": best[5],
        "fn": best[6],
        "n_gt": n_gt,
        "records": records,
    }


def train_detector(coco_path):
    train_set = DetSet("train", True)
    val_set = DetSet("val", False)
    train_loader = DataLoader(
        train_set,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )
    log(f"det train {len(train_set)} val {len(val_set)}")
    model = build_detector(coco_path)
    model.to(DEVICE)
    backbone, head = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone") or name.startswith("rpn"):
            backbone.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD(
        [
            {"params": backbone, "lr": 0.001},
            {"params": head, "lr": 0.005},
        ],
        momentum=0.9,
        weight_decay=0.0005,
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    best_ap = -1.0
    best_path = RUN / "detector_best.pth"
    best_metrics = None
    for epoch in range(DET_EPOCHS):
        if epoch == 8:
            base_lrs = [lr * 0.1 for lr in base_lrs]
        model.train()
        t0 = time.time()
        running = 0.0
        steps = 0
        for step, (images, targets) in enumerate(train_loader):
            scale = (step + 1) / 100.0 if epoch == 0 and step < 100 else 1.0
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * scale
            images = [image.to(DEVICE) for image in images]
            moved, _classes = move_targets(targets, keep_cls=False)
            loss_dict = model(images, moved)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach())
            steps += 1
            if step % 200 == 0:
                log(
                    f"det {epoch + 1}/{DET_EPOCHS} step {step}/{len(train_loader)} "
                    f"loss {running / steps:.4f}"
                )
        metrics = evaluate(model, val_loader)
        log(
            f"EPOCH {epoch + 1} loss {running / max(steps, 1):.4f} "
            f"P {metrics['precision']:.4f} R {metrics['recall']:.4f} "
            f"F1 {metrics['f1']:.4f} mAP50 {metrics['ap50']:.4f} "
            f"mAP50-95 {metrics['ap5095']:.4f} thr {metrics['thr']:.2f} "
            f"sec {time.time() - t0:.0f}"
        )
        if metrics["ap50"] > best_ap:
            best_ap = metrics["ap50"]
            best_metrics = {k: v for k, v in metrics.items() if k != "records"}
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "metrics": best_metrics}, best_path)
            log(f"saved detector epoch {epoch + 1} mAP50 {best_ap:.4f}")
    model.load_state_dict(torch.load(best_path, map_location="cpu")["model"])
    model.to(DEVICE)
    final = evaluate(model, val_loader)
    kept = {k: v for k, v in final.items() if k != "records"}
    log(
        "DETECTOR_DONE "
        f"epoch {torch.load(best_path, map_location='cpu')['epoch']} "
        f"P {kept['precision']:.4f} R {kept['recall']:.4f} F1 {kept['f1']:.4f} "
        f"mAP50 {kept['ap50']:.4f} mAP50-95 {kept['ap5095']:.4f} thr {kept['thr']:.2f}"
    )
    (RUN / "detector_metrics.txt").write_text(
        "\n".join(f"{k} {kept[k]}" for k in kept) + "\n", encoding="utf-8"
    )
    return model, kept


class CropSet(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, box, cls = self.items[index]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = min(max(x1, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        x2 = min(max(x2, x1 + 1), width)
        y2 = min(max(y2, y1 + 1), height)
        crop = image.crop((x1, y1, x2, y2))
        return self.transform(crop), cls


def collect_crops(split):
    items = []
    for ip, lp in pairs(split):
        with Image.open(ip) as image:
            width, height = image.size
        boxes, classes = read_yolo(lp, width, height)
        for box, cls in zip(boxes, classes):
            items.append((str(ip), box, cls))
    return items


def run_classifier(model, loader):
    model.eval()
    correct = 0
    total = 0
    confusion = np.zeros((4, 4), dtype=np.int64)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            pred = model(images).argmax(1)
            correct += int((pred == labels).sum())
            total += int(labels.numel())
            for gt, pr in zip(labels.cpu().numpy(), pred.cpu().numpy()):
                confusion[int(gt), int(pr)] += 1
    model.train()
    return correct / max(total, 1), total, confusion


def train_classifier(weight_path):
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_items = collect_crops("train")
    val_items = collect_crops("val")
    counts = np.bincount([c for _p, _b, c in train_items], minlength=4).astype(np.float64)
    log(f"crops train {len(train_items)} val {len(val_items)} counts {counts.tolist()}")
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1.0) / 4.0, dtype=torch.float32, device=DEVICE)
    train_loader = DataLoader(CropSet(train_items, train_tf), batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(CropSet(val_items, val_tf), batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    model = resnet18(pretrained=False)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if not n.startswith("fc")], "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_acc = -1.0
    stale = 0
    best_path = RUN / "classifier_best.pth"
    best_confusion = None
    for epoch in range(CLS_EPOCHS):
        if epoch == 8:
            for group in optimizer.param_groups:
                group["lr"] *= 0.1
        model.train()
        running = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * labels.size(0)
            seen += int(labels.size(0))
        acc, total, confusion = run_classifier(model, val_loader)
        log(f"CLS {epoch + 1}/{CLS_EPOCHS} loss {running / max(seen, 1):.4f} acc {acc:.4f} n {total}")
        if acc > best_acc:
            best_acc = acc
            stale = 0
            best_confusion = confusion
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "acc": acc}, best_path)
        else:
            stale += 1
            if stale >= CLS_PATIENCE and epoch >= 8:
                log(f"cls early stop at epoch {epoch + 1}")
                break
    state = torch.load(best_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(DEVICE)
    acc, total, confusion = run_classifier(model, val_loader)
    lines = [f"accuracy {acc:.4f}", f"n {total}", f"epoch {state['epoch']}"]
    precisions = []
    for i, name in enumerate(NAMES):
        support = int(confusion[i].sum())
        predicted = int(confusion[:, i].sum())
        hit = int(confusion[i, i])
        precision = hit / predicted if predicted else 0.0
        recall = hit / support if support else 0.0
        precisions.append(precision)
        lines.append(f"{name} precision {precision:.4f} recall {recall:.4f} support {support}")
    lines.append(f"macro_precision {float(np.mean(precisions)):.4f}")
    text = "\n".join(lines)
    log("CLASSIFIER_DONE\n" + text)
    (RUN / "classifier_metrics.txt").write_text(text + "\n", encoding="utf-8")
    return model, acc, confusion


def clip_box(box, width, height):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = min(max(x1, 0), max(width - 1, 0))
    y1 = min(max(y1, 0), max(height - 1, 0))
    x2 = min(max(x2, x1 + 1), width)
    y2 = min(max(y2, y1 + 1), height)
    return x1, y1, x2, y2


@torch.no_grad()
def pipeline(detector, classifier, thr):
    detector.eval()
    classifier.eval()
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    n_gt = matched = class_ok = n_pred = 0
    for ip, lp in pairs("val"):
        image = Image.open(ip).convert("RGB")
        width, height = image.size
        gt_boxes, gt_cls = read_yolo(lp, width, height)
        gt_boxes = np.array(gt_boxes, dtype=np.float64).reshape(-1, 4)
        gt_cls = np.array(gt_cls, dtype=np.int64)
        tensor = transforms.functional.to_tensor(image).to(DEVICE)
        output = detector([tensor])[0]
        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        keep = scores >= thr
        boxes = boxes[keep]
        scores = scores[keep]
        n_pred += len(boxes)
        n_gt += len(gt_boxes)
        if len(boxes) == 0:
            continue
        crops = []
        valid = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = clip_box(box, width, height)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crops.append(val_tf(image.crop((x1, y1, x2, y2))))
            valid.append(i)
        pred_cls = np.full(len(boxes), -1, dtype=np.int64)
        if crops:
            logits = classifier(torch.stack(crops).to(DEVICE))
            got = logits.argmax(1).cpu().numpy()
            for i, cls in zip(valid, got):
                pred_cls[i] = int(cls)
        taken = set()
        ious = box_iou(boxes, gt_boxes)
        for pi in np.argsort(-scores):
            if len(gt_boxes) == 0:
                break
            row = ious[pi].copy()
            for g in taken:
                row[g] = -1.0
            g = int(row.argmax())
            if row[g] >= IOU_MATCH:
                taken.add(g)
                matched += 1
                if pred_cls[pi] == int(gt_cls[g]):
                    class_ok += 1
    text = "\n".join(
        [
            f"gt {n_gt}",
            f"predictions {n_pred}",
            f"matched {matched}",
            f"box_recall {matched / max(n_gt, 1):.4f}",
            f"class_acc_on_matched {class_ok / max(matched, 1):.4f}",
            f"joint_recall {class_ok / max(n_gt, 1):.4f}",
            f"joint_precision {class_ok / max(n_pred, 1):.4f}",
            f"thr {thr:.2f}",
        ]
    )
    log("PIPELINE_DONE\n" + text)
    (RUN / "pipeline_metrics.txt").write_text(text + "\n", encoding="utf-8")


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True
    RUN.mkdir(parents=True, exist_ok=True)
    train_paths = {p.resolve() for p, _l in pairs("train")}
    val_paths = {p.resolve() for p, _l in pairs("val")}
    log(f"images train {len(train_paths)} val {len(val_paths)} overlap {len(train_paths & val_paths)}")
    download(
        "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        WEIGHTS / "resnet18.pth",
        20_000_000,
    )
    download(
        "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        WEIGHTS / "fasterrcnn_resnet50_fpn_coco.pth",
        167_000_000,
    )
    classifier, _acc, _confusion = train_classifier(WEIGHTS / "resnet18.pth")
    torch.cuda.empty_cache()
    detector, det_metrics = train_detector(WEIGHTS / "fasterrcnn_resnet50_fpn_coco.pth")
    pipeline(detector, classifier, det_metrics["thr"])
    log("ALL_DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
