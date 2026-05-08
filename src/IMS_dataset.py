"""
IMS_dataset.py
==============
PyTorch :class:`~torch.utils.data.Dataset` for GC-IMS spectra stored in an
HDF5 cache produced by :mod:`GANpreprocess`.

The entire cache is loaded into RAM at construction time.  This is safe and
efficient because the preprocessed cache is typically ~80 MB, and it avoids
repeated HDF5 open/close overhead during training.
"""

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class IMSDataset(Dataset):
    """
    In-memory dataset of preprocessed GC-IMS spectra.

    Loads the entire HDF5 cache into RAM at initialisation.
    Each item is a dictionary with keys ``"x"`` (spectrum tensor) and
    ``"cond"`` (condition vector).

    Parameters
    ----------
    cache_path : str
        Path to the HDF5 cache file produced by :mod:`GANpreprocess`.
    split : {"train", "val"}
        Dataset split to load.  Samples whose ``batch`` attribute equals
        *val_batch* are reserved for validation; all others are used for
        training.
    val_batch : int, optional
        Batch number held out as the validation set.  Default is ``4``.

    Raises
    ------
    RuntimeError
        If no samples are found for the requested *split* and *val_batch*
        combination.

    Examples
    --------
    >>> ds = IMSDataset("ims_cache.h5", split="train", val_batch=4)
    >>> sample = ds[0]
    >>> sample["x"].shape   # (1, H, W)
    torch.Size([1, 690, 128])
    >>> sample["cond"].shape  # (5,)
    torch.Size([5])
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
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return a single sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        dict
            A dictionary with keys:

            - ``"x"`` : :class:`torch.Tensor` of shape ``(1, H, W)`` —
              normalised spectrum in ``[0, 1]``.
            - ``"cond"`` : :class:`torch.Tensor` of shape ``(5,)`` —
              condition vector ``[lb, ec, sc, pf, time_norm]``.
        """
        return self.samples[idx]