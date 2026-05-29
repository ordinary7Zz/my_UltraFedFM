import torch 
from collections.abc import Mapping

path = r"output_dir/pretrained_ultrafedfm/log_2024-07-16_13_53_08/checkpoint.pth"

ckpt = torch.load(path, map_location="cpu")

print("type:", type(ckpt))
if isinstance(ckpt, Mapping):
    keys = list(ckpt.keys())
    print("top-level keys:", keys)

    for k in ["model", "state_dict", "module", "optimizer", "epoch", "scaler", "args"]:
        if k in ckpt:
            v = ckpt[k]
            if isinstance(v, Mapping):
                print(f"{k}: dict with {len(v)} keys")
                print(f"  first 10 keys: {list(v.keys())[:10]}")
            else:
                print(f"{k}: {type(v)} -> {v if k == 'epoch' else ''}")
else:
    print("checkpoint is not a dict-like object")