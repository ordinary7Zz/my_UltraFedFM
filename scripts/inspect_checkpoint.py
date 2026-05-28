import argparse
import os
from typing import Any

import torch


def summarize_mapping(name: str, value: Any, max_keys: int = 30) -> None:
    if not isinstance(value, dict):
        print(f"{name}: {type(value)}")
        return

    keys = list(value.keys())
    print(f"{name}: dict with {len(keys)} keys")
    print(f"{name} keys (first {min(len(keys), max_keys)}):")
    for key in keys[:max_keys]:
        item = value[key]
        if hasattr(item, "shape"):
            print(f"  - {key}: tensor shape={tuple(item.shape)} dtype={getattr(item, 'dtype', None)}")
        else:
            print(f"  - {key}: {type(item)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a PyTorch checkpoint file")
    parser.add_argument("checkpoint", help="Path to .pth/.pt checkpoint")
    args = parser.parse_args()

    ckpt = args.checkpoint
    print(f"path: {ckpt}")
    print(f"exists: {os.path.exists(ckpt)}")
    if not os.path.exists(ckpt):
        return

    print(f"size_bytes: {os.path.getsize(ckpt)}")

    try:
        obj = torch.load(ckpt, map_location="cpu")
    except Exception as exc:
        print(f"load_error: {repr(exc)}")
        return

    print(f"load_type: {type(obj)}")

    if isinstance(obj, dict):
        top_keys = list(obj.keys())
        print(f"top_level_keys ({len(top_keys)}): {top_keys[:50]}")

        for key in ["state_dict", "model", "model_state_dict", "module", "checkpoint", "encoder", "decoder"]:
            if key in obj:
                summarize_mapping(f"nested[{key}]", obj[key])

        tensor_keys = [k for k, v in obj.items() if hasattr(v, "shape")]
        if tensor_keys:
            print(f"raw_tensor_keys ({len(tensor_keys)}):")
            for key in tensor_keys[:50]:
                tensor = obj[key]
                print(f"  - {key}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")

    elif isinstance(obj, (list, tuple)):
        print(f"sequence_len: {len(obj)}")
        for idx, item in enumerate(obj[:10]):
            print(f"  [{idx}]: {type(item)}")
    else:
        print(f"value_repr: {repr(obj)[:500]}")


if __name__ == "__main__":
    main()
