#!/usr/bin/env python3
"""Resumable MVSS-Net inference preserving the official 512px/BGR preprocessing."""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
MVSS = REPO / "external/MVSS-Net"
sys.path.insert(0, str(MVSS))
from models.mvssnet import get_mvss  # noqa: E402


def load_model(device):
    model = get_mvss(backbone="resnet50", pretrained_base=False, nclass=1,
                     sobel=True, constrain=True, n_input=3)
    checkpoint = torch.load(MVSS / "ckpt/mvssnet_casia.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint, strict=True)
    return model.to(device).eval()


def preprocess(path):
    with Image.open(path) as source:
        rgb = source.convert("RGB"); original_size = rgb.size
        arr = np.asarray(rgb.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32)
    bgr = arr[..., ::-1].copy() / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy(((bgr - mean) / std).transpose(2, 0, 1)).unsqueeze(0)
    return tensor, original_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    paths = sorted(Path().glob(args.input)) if not Path(args.input).is_absolute() else sorted(
        Path(args.input).parent.glob(Path(args.input).name))
    if not paths: raise FileNotFoundError(args.input)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model = load_model(device); completed = skipped = failed = 0
    with torch.no_grad():
        for index, path in enumerate(paths, 1):
            target = out / (path.name + ".npz")
            if target.is_file(): skipped += 1; continue
            try:
                tensor, (width, height) = preprocess(path)
                _, logits = model(tensor.to(device))
                probability = torch.sigmoid(logits)
                score = float(probability.max().item())
                restored = F.interpolate(probability, size=(height, width), mode="bilinear",
                                         align_corners=False)[0, 0].cpu().numpy().astype(np.float32)
                np.savez_compressed(target, map=restored, score=np.float32(score),
                                    imgsize=np.array([height, width], dtype=np.int32))
                completed += 1
            except Exception as exc:
                failed += 1; print(f"FAILED\t{path}\t{exc}", flush=True)
            if index % 25 == 0: print(f"progress={index}/{len(paths)} completed={completed} skipped={skipped} failed={failed}", flush=True)
    print(f"done total={len(paths)} completed={completed} skipped={skipped} failed={failed}")
    if failed: raise SystemExit(2)


if __name__ == "__main__": main()
