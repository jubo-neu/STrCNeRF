import argparse
from strcnerf.utils import load_config, device_for
from strcnerf.engine import train


def main():
    parser = argparse.ArgumentParser(description="Train STrCNeRF")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--data-root")
    parser.add_argument("--output")
    args = parser.parse_args(); config = load_config(args.config)
    if args.steps is not None: config["train"]["steps"] = args.steps
    if args.data_root: config["data"]["root"] = args.data_root
    if args.output: config["train"]["output"] = args.output
    print(f"Checkpoint: {train(config, device_for(args.device), args.resume)}")


if __name__ == "__main__": main()
