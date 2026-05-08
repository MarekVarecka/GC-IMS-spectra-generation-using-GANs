"""
Z_make_synthetic.py
===================
Generate synthetic GC-IMS spectra from a trained GAN checkpoint and write
them to an HDF5 cache that is format-compatible with the real ``ims_cache.h5``.

Supports all four model variants and both culture types:

- **Pure** — single organism, one-hot ``org_vec``, e.g. ``[1, 0, 0, 0]``.
- **Mixed** — two organisms, multi-hot ``org_vec``, e.g. ``[1, 1, 0, 0]``
  (all :math:`\binom{4}{2} = 6` pairs).

GAN variants
------------
``cwgan_gp``
    Conditional WGAN + Gradient Penalty (default).
``cwgan``
    Conditional WGAN + Weight Clipping.
``cgan``
    Conditional GAN + BCE loss.
``wgan_gp``
    Unconditional WGAN + Gradient Penalty.

Usage examples
--------------
.. code-block:: bash

    # Pure cultures only (default):
    python Z_make_synthetic.py \\
        --checkpoint checkpoints/cwgan_gp_final.pt --model cwgan_gp

    # Include mixed cultures (all 6 two-organism pairs):
    python Z_make_synthetic.py \\
        --checkpoint checkpoints/cwgan_gp_final.pt --model cwgan_gp --mixed

    # Mixed cultures only:
    python Z_make_synthetic.py \\
        --checkpoint checkpoints/cwgan_gp_final.pt --model cwgan_gp \\
        --mixed --no-pure

    # Custom output path and sample count:
    python Z_make_synthetic.py \\
        --checkpoint checkpoints/cgan_final.pt --model cgan \\
        --output synthetic_cgan.h5 --n-per-cond 50 --mixed
"""

import argparse
import itertools
import sys
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import torch

from IMS_models import Generator

# ── Constants ────────────────────────────────────────────────────────────────

ORGANISMS = ["lb", "ec", "sc", "pf"]
"""Known organism codes in fixed index order."""

COND_DIM = 5
"""Condition vector dimensionality: ``[lb, ec, sc, pf, time_norm]``."""

CONDITIONAL_MODELS   = {"cwgan_gp", "cwgan", "cgan"}
"""Set of model variants that accept a condition vector."""

UNCONDITIONAL_MODELS = {"wgan_gp"}
"""Set of model variants that ignore the condition vector."""

ALL_MODELS = CONDITIONAL_MODELS | UNCONDITIONAL_MODELS
"""Union of all supported model variant identifiers."""


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: ``checkpoint``, ``model``,
        ``output``, ``real_cache``, ``n_per_cond``, ``max_hours``,
        ``mixed``, ``no_pure``, ``device``, ``seed``.

    Raises
    ------
    SystemExit
        If ``--no-pure`` is used without ``--mixed``.
    """
    parser = argparse.ArgumentParser(
        prog="Z_make_synthetic_cache.py",
        description="Generate synthetic GC-IMS spectra (pure + optional mixed cultures).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    req = parser.add_argument_group("required")
    req.add_argument("--checkpoint", "-c", metavar="PATH", required=True,
                     help="Generator checkpoint (.pt)")
    req.add_argument("--model", "-m", metavar="VARIANT", required=True,
                     choices=sorted(ALL_MODELS),
                     help=f"GAN variant. Choices: {sorted(ALL_MODELS)}")

    opt = parser.add_argument_group("optional")
    opt.add_argument("--output", "-o", metavar="PATH", default=None,
                     help="Output HDF5 path  (default: synthetic_<model>.h5)")
    opt.add_argument("--real-cache", metavar="PATH", default="ims_cache.h5",
                     help="Real HDF5 cache for rettime/drifttime axes  (default: ims_cache.h5)")
    opt.add_argument("--n-per-cond", metavar="N", type=int, default=20,
                     help="Synthetic samples per condition cell  (default: 20)")
    opt.add_argument("--max-hours", metavar="N", type=int, default=8,
                     help="Number of fermentation time steps 0…N-1  (default: 8)")
    opt.add_argument("--mixed", action="store_true",
                     help="Also generate mixed-culture conditions (all 6 two-organism pairs)")
    opt.add_argument("--no-pure", action="store_true",
                     help="Skip pure single-organism conditions (requires --mixed)")
    opt.add_argument("--device", metavar="STR", default="auto",
                     choices=["auto", "cuda", "cpu"],
                     help='"auto" picks CUDA when available  (default: auto)')
    opt.add_argument("--seed", metavar="INT", type=int, default=42,
                     help="Global RNG seed  (default: 42)")

    args = parser.parse_args()

    if args.no_pure and not args.mixed:
        parser.error("--no-pure requires --mixed")

    return args


# ── Condition builder ─────────────────────────────────────────────────────────

def build_conditions(
    include_pure: bool,
    include_mixed: bool,
    max_hours: int,
) -> list[tuple[np.ndarray, str]]:
    """
    Build a list of ``(org_vec, label)`` tuples for all requested culture types.

    Generates one entry per unique organism combination; hours are handled
    separately inside :func:`generate`.

    Parameters
    ----------
    include_pure : bool
        If ``True``, include one-hot single-organism conditions.
    include_mixed : bool
        If ``True``, include all :math:`\binom{4}{2} = 6` two-organism
        multi-hot conditions.
    max_hours : int
        Not used directly; kept for API symmetry with the generation loop.

    Returns
    -------
    list of tuple
        Each element is ``(org_vec, culture_label)`` where:

        - ``org_vec`` : :class:`numpy.ndarray` of shape ``(4,)``, ``float32``.
        - ``culture_label`` : ``str``, e.g. ``"lb"`` (pure) or ``"ec+lb"`` (mixed).

    Examples
    --------
    >>> conds = build_conditions(include_pure=True, include_mixed=False, max_hours=8)
    >>> [label for _, label in conds]
    ['lb', 'ec', 'sc', 'pf']
    """
    conditions = []

    if include_pure:
        for i, org in enumerate(ORGANISMS):
            vec = np.zeros(4, dtype=np.float32)
            vec[i] = 1.0
            conditions.append((vec, org))

    if include_mixed:
        # 2-organism pairs
        for combo in itertools.combinations(range(len(ORGANISMS)), 2):
            vec = np.zeros(4, dtype=np.float32)
            for idx in combo:
                vec[idx] = 1.0
            label = "+".join(sorted(ORGANISMS[i] for i in combo))
            conditions.append((vec, label))

    return conditions


# ── Generator loading ─────────────────────────────────────────────────────────

def resolve_device(choice: str) -> torch.device:
    """
    Resolve the target compute device.

    Parameters
    ----------
    choice : {"auto", "cuda", "cpu"}
        Device selection string.  ``"auto"`` picks CUDA if available.

    Returns
    -------
    torch.device
        Resolved device object.
    """
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def load_generator(checkpoint: str, device: torch.device) -> Generator:
    """
    Restore a :class:`~IMS_models.Generator` from a checkpoint file.

    Parameters
    ----------
    checkpoint : str
        Path to the ``.pt`` checkpoint file saved by :mod:`IMS_train_models`.
        Required keys: ``G_state``, ``img_h``, ``img_w``.
        Optional keys: ``cond_dim`` (default ``5``), ``z_dim`` (default ``128``).
    device : torch.device
        Target device for the loaded model.

    Returns
    -------
    Generator
        Generator in evaluation mode (``model.eval()`` called).
    """
    ckpt     = torch.load(checkpoint, map_location=device)
    img_h    = int(ckpt["img_h"])
    img_w    = int(ckpt["img_w"])
    cond_dim = int(ckpt.get("cond_dim", COND_DIM))
    z_dim    = int(ckpt.get("z_dim",   128))

    G = Generator(z_dim=z_dim, cond_dim=cond_dim,
                  img_h=img_h, img_w=img_w).to(device)
    G.load_state_dict(ckpt["G_state"])
    G.eval()
    return G


def get_reference_axes(real_cache: str) -> tuple:
    """
    Read retention-time and drift-time axes from the real HDF5 cache.

    These axes are copied verbatim into the synthetic cache so that the two
    datasets remain axis-aligned for downstream comparison.

    Parameters
    ----------
    real_cache : str
        Path to the real ``ims_cache.h5`` file.

    Returns
    -------
    rettime : numpy.ndarray
        Retention-time axis array from the first sample in the cache.
    drifttime : numpy.ndarray
        Drift-time axis array from the first sample in the cache.
    """
    with h5py.File(real_cache, "r") as hf:
        k = sorted(hf.keys())[0]
        return hf[k]["rettime"][:], hf[k]["drifttime"][:]


# ── Writing ───────────────────────────────────────────────────────────────────

def _write_sample(hf, count, arr, rettime, drifttime,
                  org_vec, culture_label, hour, time_norm, replica):
    """
    Write a single synthetic spectrum into an open HDF5 file.

    Parameters
    ----------
    hf : h5py.File
        Open HDF5 file object in write mode.
    count : int
        Global sample counter used to generate the group key
        (``sample_NNNN``).
    arr : numpy.ndarray
        Spectrum values array of shape ``(H, W)``.
    rettime : numpy.ndarray
        Retention-time axis to store alongside the spectrum.
    drifttime : numpy.ndarray
        Drift-time axis to store alongside the spectrum.
    org_vec : numpy.ndarray
        Multi-hot organism vector of shape ``(4,)``.
    culture_label : str
        Human-readable culture label (e.g. ``"lb"`` or ``"ec+lb"``).
    hour : int
        Fermentation hour index.
    time_norm : float
        Normalised fermentation time ``hour / (max_hours - 1)``.
    replica : int
        Zero-based replica index within this ``(condition, hour)`` cell.
    """
    key = f"sample_{count:04d}"
    grp = hf.create_group(key)
    grp.create_dataset("values",    data=arr.astype(np.float32), compression="lzf")
    grp.create_dataset("rettime",   data=rettime)
    grp.create_dataset("drifttime", data=drifttime)
    grp.create_dataset("org_vec",   data=org_vec)
    grp.attrs["name"]         = f"syn_{culture_label}_t{hour}_r{replica}"
    grp.attrs["hour"]         = int(hour)
    grp.attrs["time_norm"]    = float(time_norm)
    grp.attrs["culture_type"] = "mixed" if "+" in culture_label else "pure"
    grp.attrs["culture_label"]= culture_label
    grp.attrs["batch"]        = 0
    grp.attrs["batch_id"]     = f"synthetic_{culture_label}"


# ── Generation ────────────────────────────────────────────────────────────────

def generate(G, hf_out, rettime, drifttime, conditions, max_hours,
             n_per_cond, device, unconditional=False) -> int:
    """
    Iterate over all ``(condition × hour)`` cells and write spectra to HDF5.

    For conditional models the full condition vector ``[org_vec | time_norm]``
    is fed to the generator.  For unconditional models (``wgan_gp``) only a
    noise vector is fed; the organism metadata is still written as HDF5
    attributes for bookkeeping.

    Parameters
    ----------
    G : Generator
        Generator network in evaluation mode.
    hf_out : h5py.File
        Open HDF5 file in write mode.
    rettime : numpy.ndarray
        Retention-time axis copied from the real cache.
    drifttime : numpy.ndarray
        Drift-time axis copied from the real cache.
    conditions : list of tuple
        Output of :func:`build_conditions` —
        ``[(org_vec, culture_label), ...]``.
    max_hours : int
        Number of fermentation time steps (``0`` to ``max_hours - 1``).
    n_per_cond : int
        Number of synthetic spectra generated per ``(condition, hour)`` cell.
    device : torch.device
        Compute device.
    unconditional : bool, optional
        If ``True``, no condition vector is passed to the generator.
        Default is ``False``.

    Returns
    -------
    int
        Total number of spectra written to *hf_out*.
    """
    count = 0
    with torch.no_grad():
        for org_vec, culture_label in conditions:
            for hour in range(max_hours):
                time_norm = hour / max(max_hours - 1, 1)

                if unconditional:
                    # wgan_gp: no condition fed to model; org_vec is metadata only
                    z    = torch.randn(n_per_cond, G.z_dim, device=device)
                    fake = G(z, cond=None).cpu().squeeze(1).numpy()
                else:
                    cond_row = torch.tensor(
                        org_vec.tolist() + [time_norm],
                        dtype=torch.float32, device=device,
                    ).unsqueeze(0).expand(n_per_cond, -1)
                    z    = torch.randn(n_per_cond, G.z_dim, device=device)
                    fake = G(z, cond_row).cpu().squeeze(1).numpy()

                for i, arr in enumerate(fake):
                    _write_sample(hf_out, count, arr, rettime, drifttime,
                                  org_vec, culture_label, hour, time_norm, i)
                    count += 1

            culture_type = "mixed" if "+" in culture_label else "pure"
            print(f"  [{culture_type:<5}  {culture_label:<12}]  "
                  f"{max_hours} hours x {n_per_cond} = "
                  f"{max_hours * n_per_cond} spectra written  "
                  f"(total: {count})")
    return count


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point: parse arguments, load the generator, and write the synthetic cache.

    Reads the trained generator checkpoint, constructs all requested
    condition combinations, generates *n_per_cond* spectra per
    ``(condition, hour)`` cell, and saves everything to an HDF5 file whose
    structure mirrors the real ``ims_cache.h5``.
    """
    args = parse_args()

    out_path = args.output or f"synthetic_{args.model}.h5"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── banner ─────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  Z_make_synthetic_cache")
    print(f"  {datetime.now():%Y-%m-%d  %H:%M:%S}")
    print(f"{'═'*62}")
    print(f"  Model variant  : {args.model}")
    print(f"  Checkpoint     : {args.checkpoint}")
    print(f"  Real cache     : {args.real_cache}")
    print(f"  Output         : {out_path}")
    print(f"  N per cond     : {args.n_per_cond}")
    print(f"  Max hours      : {args.max_hours}")
    print(f"  Pure cultures  : {not args.no_pure}")
    print(f"  Mixed cultures : {args.mixed}")
    print(f"  Seed           : {args.seed}")

    # ── validate ───────────────────────────────────────────────────────
    if not Path(args.checkpoint).exists():
        sys.exit(f"\n    Checkpoint not found: {args.checkpoint}")
    if not Path(args.real_cache).exists():
        sys.exit(f"\n    Real cache not found: {args.real_cache}")
    if Path(out_path).exists():
        print(f"\n    Output already exists and will be overwritten: {out_path}")

    # ── build condition list ────────────────────────────────────────────
    conditions = build_conditions(
        include_pure  = not args.no_pure,
        include_mixed = args.mixed,
        max_hours     = args.max_hours,
    )

    pure_conds  = [(v, l) for v, l in conditions if "+" not in l]
    mixed_conds = [(v, l) for v, l in conditions if "+" in l]

    print(f"\n  Conditions:")
    print(f"    Pure  : {len(pure_conds):2d}  {[l for _,l in pure_conds]}")
    print(f"    Mixed : {len(mixed_conds):2d}  {[l for _,l in mixed_conds]}")
    print(f"    Total : {len(conditions)} combos x {args.max_hours} hours "
          f"x {args.n_per_cond} = "
          f"{len(conditions) * args.max_hours * args.n_per_cond} spectra expected")

    # ── setup ──────────────────────────────────────────────────────────
    device = resolve_device(args.device)
    print(f"\n  Device   : {device}")

    print(f"  Loading generator …")
    G = load_generator(args.checkpoint, device)
    print(f"  Generator: {G.img_h}x{G.img_w}  z_dim={G.z_dim}  cond_dim={G.cond_dim}\n")

    rettime, drifttime = get_reference_axes(args.real_cache)
    unconditional = args.model in UNCONDITIONAL_MODELS

    if unconditional:
        print(f"Unconditional model ({args.model}): org_vec is metadata only.")

    # ── generate ───────────────────────────────────────────────────────
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as hf_out:
        hf_out.attrs["model_variant"]   = args.model
        hf_out.attrs["checkpoint"]      = str(args.checkpoint)
        hf_out.attrs["n_per_cond"]      = args.n_per_cond
        hf_out.attrs["max_hours"]       = args.max_hours
        hf_out.attrs["include_pure"]    = not args.no_pure
        hf_out.attrs["include_mixed"]   = args.mixed
        hf_out.attrs["seed"]            = args.seed
        hf_out.attrs["created"]         = datetime.now().isoformat()
        hf_out.attrs["height"]          = G.img_h
        hf_out.attrs["width"]           = G.img_w

        count = generate(
            G, hf_out, rettime, drifttime,
            conditions, args.max_hours,
            args.n_per_cond, device,
            unconditional=unconditional,
        )

    print(f"\n{'─'*62}")
    print(f"Saved {count} synthetic spectra to {out_path}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
