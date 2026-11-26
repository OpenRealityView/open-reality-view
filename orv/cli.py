"""Command line: ``orv info`` prints the environment, ``orv smoke`` runs a toy forward pass."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import random
import time

import torch
import yaml

from orv import __version__
from orv.data import synthetic_sample
from orv.geometry import ChaseCameraFamily
from orv.models import OpenRealityView


def info(_: argparse.Namespace) -> None:
    print(f"open-reality-view {__version__}")
    print(f"python {platform.python_version()} | torch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"cuda {torch.version.cuda} | {torch.cuda.get_device_name(0)}")
    else:
        print("cuda not available")
    try:
        print(f"gsplat {importlib.metadata.version('gsplat')}")
    except importlib.metadata.PackageNotFoundError:
        print("gsplat not installed: only the naive CPU renderer is available")


def smoke(args: argparse.Namespace) -> None:
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    model = OpenRealityView(**cfg["model"]).to(device).eval()
    sample = synthetic_sample(**cfg["input"]).to(device)
    out = cfg["output"]
    family = ChaseCameraFamily(**cfg.get("virtual_camera", {}))
    views = family.sample(
        sample.ego_to_world[:, -1], out["views"], out["width"], out["height"], random.Random(0)
    )
    with torch.no_grad():
        start = time.perf_counter()
        gaussians = model.lifter(sample)
        first = model.render_views(gaussians, views)
        second = model.render_views(gaussians, views, prev=first)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    print(f"device {device} | renderer {model.renderer}")
    print(f"gaussians {tuple(gaussians.means.shape)}")
    print(f"render 1/{model.render_scale} {tuple(second.render.features.shape)}")
    print(f"output {tuple(second.rgb.shape)} | {elapsed * 1e3:.0f} ms for lift + 2 decoded steps")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="orv", description="Open Reality View")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="print the environment").set_defaults(func=info)
    p = sub.add_parser("smoke", help="toy forward pass on synthetic inputs")
    p.add_argument("--config", default="configs/debug.yaml")
    p.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    p.set_defaults(func=smoke)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
