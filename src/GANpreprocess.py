"""
GANpreprocess.py
================
Preprocessing pipeline for raw GC-IMS ``.mea`` spectra.

Reads raw measurement files via :func:`~meaLoader.load_mea_files`, applies a
multi-step signal processing pipeline to each spectrum, and writes the results
to a compressed HDF5 cache file ready for GAN training.

Pipeline steps (Christmann et al. 2024 / Kirtsanis et al. 2025)
----------------------------------------------------------------
1. RIP-relative drift-time alignment — ``spec.riprel()``
2. Wavelet compression — ``spec.wavecompr(wavelet="db3", level=3)``
3. Retention-time resampling — ``spec.resample(7)``
4. Top-hat baseline correction — ``spec.tophat(size=15)``
5. Drift-time cropping — removes RIP, keeps analyte zone ``[1.05, 2.10]`` ms
6. Retention-time cropping — keeps VOC elution window ``[70, 780]`` s
7. Drift-time interpolation — resamples to a fixed width of 128 points
8. RIP-based normalisation — scales intensities to ``[0, 1]``

Usage
-----
.. code-block:: bash

    python GANpreprocess.py --root /data/IMS --cache ims_cache.h5
"""

import re
import argparse
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from ims import Spectrum as IMSSpectrum
from meaLoader import load_mea_files

# ── Organism registry ─────────────────────────────────────────────────────────
ORGANISMS = ["lb", "ec", "sc", "pf"]
"""List of known organism codes used for multi-hot encoding."""

MAX_HOURS = 8
"""Maximum fermentation duration in hours."""

COND_DIM  = len(ORGANISMS) + 1   # 4 organism flags + 1 normalized time
"""Condition vector dimensionality: 4 organism flags + 1 normalised time."""

def parse_organisms(name: str):
    """
    Extract organism codes from a file or folder name.

    Normalises the name to lowercase with underscores, then checks for
    membership of each code in :data:`ORGANISMS`.

    Parameters
    ----------
    name : str
        Full file path or folder name to inspect.

    Returns
    -------
    list of str
        Organism codes found in *name* (e.g. ``["lb", "ec"]``).
        Returns an empty list if no known code is found.
    """
    s = name.lower().replace("-", "_").replace(" ", "_")
    found = []
    for org in ORGANISMS:
        if org in s:
            found.append(org)
    return found

def organisms_to_multihot(org_list):
    """
    Convert a list of organism codes to a multi-hot indicator vector.

    Parameters
    ----------
    org_list : list of str
        Organism codes to encode (e.g. ``["lb", "ec"]``).

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(len(ORGANISMS),)`` with ``1.0`` at each
        index corresponding to a present organism and ``0.0`` elsewhere
        (e.g. ``[1., 1., 0., 0.]``).

    Examples
    --------
    >>> organisms_to_multihot(["lb", "sc"])
    array([1., 0., 1., 0.], dtype=float32)
    """
    vec = np.zeros(len(ORGANISMS), dtype=np.float32)
    for o in org_list:
        if o in ORGANISMS:
            vec[ORGANISMS.index(o)] = 1.0
    return vec

def parse_hour_from_filename(stem: str):
    """
    Extract the fermentation hour from a filename stem.

    Looks for a pattern of the form ``t<digit>_`` (e.g. ``t0_``, ``t6_``).

    Parameters
    ----------
    stem : str
        Filename stem (without extension) to parse.

    Returns
    -------
    int or None
        Parsed hour in the range ``[0, MAX_HOURS]``, or ``None`` if the
        pattern is not found or the value is out of range.
    """
    s = stem.lower()
    time = re.search(r"t\d_", s)
    if time:
        try:
            hour = int(time.group(0)[1:-1])
            if 0 <= hour <= MAX_HOURS:
                return hour
        except ValueError:
            pass

def parse_batch_number(folder_name: str):
    """
    Extract the batch number from a folder name.

    Searches for a pattern such as ``Batch_1`` or ``batch-2`` (case-insensitive).

    Parameters
    ----------
    folder_name : str
        Name of the folder to parse (e.g. ``"210429_EC_Batch_1"``).

    Returns
    -------
    int
        Parsed batch number, or ``1`` if the pattern is not found.

    Examples
    --------
    >>> parse_batch_number("210429_EC_Batch_3")
    3
    """
    match = re.search(r"batch[_\-\s]?(\d+)", folder_name.lower())
    if match:
        return int(match.group(1))
    return 1

def scan_dataset(root: str):
    """
    Scan a dataset directory and parse metadata from file paths.

    Calls :func:`~meaLoader.load_mea_files` to obtain a flat list of
    ``.mea`` paths, then parses the organism labels, batch number, and
    fermentation hour for each file.

    Parameters
    ----------
    root : str
        Root directory passed to :func:`~meaLoader.load_mea_files`.

    Returns
    -------
    list of dict
        One record per ``.mea`` file with the following keys:

        - ``"path"`` : str — absolute file path.
        - ``"org_vec"`` : numpy.ndarray — multi-hot organism vector.
        - ``"batch_id"`` : str — unique batch identifier string.
        - ``"batch_number"`` : int — numeric batch index.
        - ``"culture_type"`` : ``"pure"`` or ``"mixed"``.
        - ``"hour"`` : int — fermentation hour (``0`` if not parseable).

    Warns
    -----
    Prints a warning to stdout for any file whose organisms or hour cannot
    be parsed.
    """
    path_list = load_mea_files(root)

    records  = []
    by_batch = defaultdict(list)

    for path in path_list:
        p = Path(path)

        # Walk path parts in reverse to find the most specific organism label
        organisms = []
        for part in reversed(p.parts):
            found = parse_organisms(part)
            if found:
                organisms = found
                break

        if not organisms:
            print(f"[WARN] Could not parse organisms from path: {path}")
            continue

        batch_id = f"{p.parent.parent.name}__{p.parent.name}"

        rec = {
            "path":         str(p),
            "org_vec":      organisms_to_multihot(organisms),
            "batch_id":     batch_id,
            "batch_number": parse_batch_number(p.parent.name),
            "culture_type": "mixed" if len(organisms) > 1 else "pure",
            "hour":         None,
        }
        records.append(rec)
        by_batch[batch_id].append(rec)

    # Resolve fermentation hours per batch
    for batch_id, batch_records in by_batch.items():
        parsed = {
            r["path"]: parse_hour_from_filename(Path(r["path"]).stem)
            for r in batch_records
        }
        if all(v is not None for v in parsed.values()):
            for r in batch_records:
                r["hour"] = parsed[r["path"]]
        else:
            print(f"[WARN] Could not parse hour for batch '{batch_id}', defaulting to 0")
            for r in batch_records:
                r["hour"] = 0

    return records

def build_cache(args):
    """
    Preprocess all spectra and write them to an HDF5 cache file.

    Scans *args.root* for ``.mea`` files, applies :func:`preprocess` to each
    spectrum, and stores the results together with metadata in an HDF5 file at
    *args.cache*.  Global statistics (``log_mean``, ``log_std``) computed over
    the log-transformed intensities are stored as root-level attributes for
    use during GAN training normalisation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.  Expected attributes:

        - ``root`` (str) — dataset root directory.
        - ``cache`` (str) — output HDF5 path.
        - ``dt_start``, ``dt_stop`` (float) — drift-time crop bounds.
        - ``rt_start``, ``rt_stop`` (float) — retention-time crop bounds.

    Raises
    ------
    RuntimeError
        If no records are found in *args.root*, or if all files fail to
        preprocess.
    """
    records = scan_dataset(args.root)

    if not records:
        raise RuntimeError(
            f"No records produced. Check that load_mea_files('{args.root}') "
            f"returns paths and that organism names are parseable."
        )

    from collections import Counter
    print(f"\nLoaded {len(records)} files via load_mea_files()")

    print("\nDry-run on first file to detect output shape...")
    try:
        test = IMSSpectrum.read_mea(records[0]["path"])
        test = preprocess(
            test,
            dt_start=args.dt_start, dt_stop=args.dt_stop,
            rt_start=args.rt_start, rt_stop=args.rt_stop,
        )
        H, W = test.values.shape
        print(f"  Output shape : ({H}, {W})")
        print(f"  Value range  : [{test.values.min():.4f}, {test.values.max():.4f}]")
    except Exception as e:
        print(f"  [ERROR] Dry-run failed: {e}")
        H, W = None, None

    valid_count     = 0
    log_accumulator = []

    with h5py.File(args.cache, "w") as hf:
        hf.attrs["cond_dim"]  = COND_DIM
        hf.attrs["max_hours"] = MAX_HOURS
        hf.attrs["dt_start"]  = args.dt_start
        hf.attrs["dt_stop"]   = args.dt_stop
        hf.attrs["rt_start"]  = args.rt_start
        hf.attrs["rt_stop"]   = args.rt_stop
        if H: hf.attrs["height"] = H
        if W: hf.attrs["width"]  = W

        for idx, rec in enumerate(tqdm(records, desc="Preprocessing")):
            try:
                spec = IMSSpectrum.read_mea(rec["path"])
                spec = preprocess(
                    spec,
                    dt_start=args.dt_start, dt_stop=args.dt_stop,
                    rt_start=args.rt_start, rt_stop=args.rt_stop,
                )

                grp = hf.create_group(f"sample_{idx:04d}")
                grp.create_dataset("values",    data=spec.values.astype(np.float32), compression="lzf")
                grp.create_dataset("rettime",   data=spec.ret_time.astype(np.float32))
                grp.create_dataset("drifttime", data=spec.drift_time.astype(np.float32))
                grp.create_dataset("org_vec",   data=rec["org_vec"])

                grp.attrs["name"]         = spec.name
                grp.attrs["batch_id"]     = rec["batch_id"]
                grp.attrs["batch"]        = rec["batch_number"]
                grp.attrs["culture_type"] = rec["culture_type"]
                grp.attrs["hour"]         = rec["hour"]
                grp.attrs["time_norm"]    = float(rec["hour"]) / MAX_HOURS

                log_accumulator.append(np.log1p(spec.values.ravel()))
                valid_count += 1

            except Exception as e:
                print(f"\n[ERROR] Skipping {rec['path']}: {e}")

        hf.attrs["n_samples"] = valid_count

        if not log_accumulator:
            raise RuntimeError("No samples were successfully processed.")

        all_log = np.concatenate(log_accumulator)
        hf.attrs["log_mean"] = float(np.mean(all_log))
        hf.attrs["log_std"]  = float(np.std(all_log))

    print(f"\nCache written : {args.cache}")
    print(f"Valid samples : {valid_count} / {len(records)}")
    with h5py.File(args.cache, "r") as hf:
        print(f"Output shape  : ({hf.attrs.get('height', '?')}, {hf.attrs.get('width', '?')})")
        print(f"log_mean      : {hf.attrs['log_mean']:.6f}")
        print(f"log_std       : {hf.attrs['log_std']:.6f}")
        print(
            f"\n train.py flags: "
            f"--height {hf.attrs.get('height', '?')} "
            f"--width {hf.attrs.get('width', '?')} "
            f"(cond_dim=5 is automatic)"
        )

def interpolate_to_fixed_width(spec, n_drift: int = 1024):
    """
    Resample the drift-time axis to exactly *n_drift* points.

    Uses linear interpolation via :func:`scipy.interpolate.interp1d`.
    Must be called **before** any compression step that changes the spectral
    width.

    Parameters
    ----------
    spec : ims.Spectrum
        Spectrum object whose ``values`` array has shape ``(n_ret, n_drift_orig)``.
    n_drift : int, optional
        Target number of drift-time points. Default is ``1024``.

    Returns
    -------
    ims.Spectrum
        The same spectrum object with ``values`` and ``drift_time`` resampled
        to *n_drift* columns.
    """
    from scipy.interpolate import interp1d

    old_x = np.linspace(0, 1, spec.values.shape[1])
    new_x = np.linspace(0, 1, n_drift)
    f = interp1d(old_x, spec.values, axis=1, kind='linear')
    spec.values   = f(new_x).astype(np.float32)
    spec.drift_time = np.interp(new_x, old_x, spec.drift_time)
    return spec

def preprocess(
    spectrum,
    tophat_sz: int   = 15,
    dt_start:  float = 1.05,
    dt_stop:   float = 2.10,
    rt_start:  float = 70.0,
    rt_stop:   float = 780.0):
    """
    Apply the full preprocessing pipeline to a single GC-IMS spectrum.

    Executes the following steps in order:

    1. **RIP-relative alignment** — aligns drift-time axis relative to the
       reactant ion peak (RIP) position.
    2. **Wavelet compression** — denoises and compresses along the drift-time
       axis using a Daubechies-3 (``db3``) wavelet at level 3.
    3. **Retention-time resampling** — reduces the number of retention-time
       rows by a factor of 7.
    4. **Top-hat baseline correction** — suppresses slowly varying background
       drift with a structuring element of size *tophat_sz*.
    5. **Drift-time cropping** — retains only ``[dt_start, dt_stop]`` ms,
       removing the RIP and pre-RIP region.
    6. **Retention-time cropping** — retains only ``[rt_start, rt_stop]`` s,
       covering the VOC elution window.
    7. **Fixed-width interpolation** — resamples to 128 drift-time points.
    8. **Normalisation** — scales all values to ``[0, 1]`` using the original
       RIP maximum intensity.

    Parameters
    ----------
    spectrum : ims.Spectrum
        Raw GC-IMS spectrum loaded from a ``.mea`` file.
    tophat_sz : int, optional
        Structuring element size for morphological top-hat correction.
        Default is ``15``.
    dt_start : float, optional
        Lower bound for drift-time cropping in milliseconds. Default is ``1.05``.
    dt_stop : float, optional
        Upper bound for drift-time cropping in milliseconds. Default is ``2.10``.
    rt_start : float, optional
        Lower bound for retention-time cropping in seconds. Default is ``70.0``.
    rt_stop : float, optional
        Upper bound for retention-time cropping in seconds. Default is ``780.0``.

    Returns
    -------
    ims.Spectrum
        Preprocessed spectrum with ``values`` of shape ``(n_ret, 128)``
        normalised to ``[0, 1]``.
    """

    spec = spectrum.copy()

    rip_max = float(spec.values.max())

    spec = spectrum.copy()
    # 1 — RIP-relative drift time alignment
    spec.riprel()
    # 2 — Wavelet compression per drift-time row (denoising + compression)
    #     direction="drifttime" → pywt.wavedec(axis=1) → each row independent
    spec.wavecompr(direction="drift_time", wavelet="db3", level=3)
    # 3 - Resample the number of rows
    spec.resample(7)
    # 3 — Top-hat baseline correction (after wavelet, on compressed data)
    spec.tophat(size=tophat_sz)
    # 4 — Cut drift time (removes RIP region, keeps analyte zone)
    spec.cut_dt(dt_start, dt_stop)
    # 5 — Cut retention time (keeps VOC elution window)
    spec.cut_rt(rt_start, rt_stop)
    # 6 — Interpolate to fixed width (fixes shape inconsistency across files)
    spec = interpolate_to_fixed_width(spec, n_drift=128)
    # 7 — RIP-based normalization to [0, 1] — LAST step
    spec.values = spec.values / (rip_max + 1e-9)
    spec.values = np.clip(spec.values, 0.0, 1.0)

    return spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",     type=str, required=True,
                        help="Root directory passed to load_mea_files()")
    parser.add_argument("--cache",    type=str, default="ims_cache.h5")
    parser.add_argument("--wavelet",  type=str, default="db3")
    parser.add_argument("--level",    type=int, default=3)
    parser.add_argument("--dt_start", type=float, default=1.05)
    parser.add_argument("--dt_stop",  type=float, default=2.10)
    parser.add_argument("--rt_start", type=float, default=70.0)
    parser.add_argument("--rt_stop",  type=float, default=780.0)
    args = parser.parse_args()
    build_cache(args)