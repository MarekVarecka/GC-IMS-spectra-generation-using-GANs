import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class IMSDataset(Dataset):
    """
    Loads entire HDF5 cache into RAM at init.
    Safe and fast because the cache is ~80 MB.
    Eliminates repeated HDF5 open/close overhead during training.
    """

    def __init__(self, cache_path: str, split: str = "train", val_batch: int = 4):
        self.samples = []

        with h5py.File(cache_path, "r") as hf:
            target_h = int(hf.attrs["height"])
            target_w = int(hf.attrs["width"])

            all_keys = sorted(k for k in hf.keys() if k.startswith("sample_"))
            for k in all_keys:
                grp   = hf[k]
                batch = int(grp.attrs["batch"])

                if split == "train" and batch == val_batch:
                    continue
                if split == "val" and batch != val_batch:
                    continue

                values    = np.array(grp["values"],  dtype=np.float32)  # (H, W)
                org_vec   = np.array(grp["org_vec"], dtype=np.float32)  # (4,)
                time_norm = float(grp.attrs["time_norm"])

                # Convert and resize once at load time — not per batch
                x = torch.from_numpy(values).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                if x.shape[2] != target_h or x.shape[3] != target_w:
                    x = F.interpolate(x, size=(target_h, target_w),
                                      mode="bilinear", align_corners=False)
                x = x.squeeze(0)                                         # (1, H, W)

                cond = torch.cat([
                    torch.from_numpy(org_vec),
                    torch.tensor([time_norm], dtype=torch.float32),
                ])                                                        # (5,)

                self.samples.append({"x": x, "cond": cond})

        if not self.samples:
            raise RuntimeError(f"No samples found for split='{split}', val_batch={val_batch}")

        print(f"[IMSDataset] Loaded {len(self.samples)} samples into RAM ({split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]