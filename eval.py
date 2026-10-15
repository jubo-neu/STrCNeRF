import argparse
import json
import torch
from strcnerf.checkpoint import model_from_checkpoint
from strcnerf.data import SceneDataset
from strcnerf.rendering import Renderer
from strcnerf.engine import evaluate
from strcnerf.utils import device_for


def main():
    parser = argparse.ArgumentParser(description="Evaluate held-out views/scenes")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--source-view", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="runs/eval")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1: parser.error("--limit must be positive")
    torch.set_num_threads(4); device = device_for(args.device)
    model, state = model_from_checkpoint(args.checkpoint, device)
    data = dict(state["config"]["data"])
    if args.data_root: data["root"] = args.data_root
    dataset = SceneDataset(**data, split=args.split, evaluation=True, source_view=args.source_view)
    result = evaluate(model, Renderer(**state["config"]["render"]), dataset, device, args.output, args.lpips, args.limit)
    print(json.dumps(result["mean_over_views"], indent=2))


if __name__ == "__main__": main()
