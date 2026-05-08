"""
IMS_train_models.py
===================
Unified training script for all four GAN ablation variants.

Each variant shares the same :class:`~IMS_models.Generator` and
:class:`~IMS_models.Critic` architecture but differs in its loss function
and Lipschitz enforcement strategy.

Model variants
--------------
``cwgan_gp``
    Conditional WGAN with Gradient Penalty — the primary model.
    Wasserstein loss + GP term (Gulrajani et al., 2017).
``cwgan``
    Conditional WGAN with weight clipping (Arjovsky et al., 2017).
    No gradient penalty.
``cgan``
    Conditional GAN with BCE loss (Mirza & Osindero, 2014).
    No Wasserstein distance.
``wgan_gp``
    Unconditional WGAN-GP.  Condition vector is ignored in both G and D.

Usage
-----
.. code-block:: bash

    python IMS_train_models.py --model cwgan_gp --cache ims_cache.h5
    python IMS_train_models.py --model cwgan    --cache ims_cache.h5
    python IMS_train_models.py --model cgan     --cache ims_cache.h5
    python IMS_train_models.py --model wgan_gp  --cache ims_cache.h5

Checkpoints are saved to ``<out_dir>/<model_name>/checkpoint_<epoch>.pt``.
The final generator weights are exported as ``generator_final.pt``.
"""

import argparse
import os

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from IMS_dataset import IMSDataset
from IMS_models import Generator, Critic, gradient_penalty


# ── Model registry ────────────────────────────────────────────────────────────

MODELS = ["cwgan_gp", "cwgan", "cgan", "wgan_gp"]
"""List of supported GAN variant identifiers."""

CLIP_VALUE = 0.01
"""Weight clipping bound used by the ``cwgan`` variant."""


# ── Training functions ────────────────────────────────────────────────────────

def train_step_cwgan_gp(G, D, opt_G, opt_D, real, cond, args, device):
    """
    Single training step for the Conditional WGAN-GP variant.

    Runs *n_critic* critic updates (Wasserstein loss + gradient penalty)
    followed by one generator update.

    Parameters
    ----------
    G : Generator
        Generator network.
    D : Critic
        Critic network.
    opt_G : torch.optim.Optimizer
        Generator optimiser.
    opt_D : torch.optim.Optimizer
        Critic optimiser.
    real : torch.Tensor
        Real spectra batch of shape ``(B, 1, H, W)``.
    cond : torch.Tensor
        Condition vectors of shape ``(B, cond_dim)``.
    args : argparse.Namespace
        Training configuration (``n_critic``, ``gp_lambda``).
    device : torch.device
        Target compute device.

    Returns
    -------
    loss_D : float
        Mean critic loss over *n_critic* updates.
    loss_G : float
        Generator loss for this step.
    """
    B = real.size(0)
    loss_D_val = 0.0

    for _ in range(args.n_critic):
        z    = G.sample_z(B, device)
        with torch.no_grad():
            fake = G(z, cond)

        gp     = gradient_penalty(D, real, fake.detach(), cond)
        loss_D = D(fake.detach(), cond).mean() - D(real, cond).mean() \
                 + args.gp_lambda * gp

        opt_D.zero_grad(set_to_none=True)
        loss_D.backward()
        opt_D.step()
        loss_D_val += loss_D.item()

    z      = G.sample_z(B, device)
    fake   = G(z, cond)
    loss_G = -D(fake, cond).mean()

    opt_G.zero_grad(set_to_none=True)
    loss_G.backward()
    opt_G.step()

    return loss_D_val / args.n_critic, loss_G.item()


def train_step_cwgan(G, D, opt_G, opt_D, real, cond, args, device):
    """
    Single training step for the Conditional WGAN (weight clipping) variant.

    Runs *n_critic* critic updates with Wasserstein loss and enforces the
    Lipschitz constraint via weight clipping to ``±CLIP_VALUE``.

    Parameters
    ----------
    G : Generator
        Generator network.
    D : Critic
        Critic network.
    opt_G : torch.optim.Optimizer
        Generator optimiser.
    opt_D : torch.optim.Optimizer
        Critic optimiser.
    real : torch.Tensor
        Real spectra batch of shape ``(B, 1, H, W)``.
    cond : torch.Tensor
        Condition vectors of shape ``(B, cond_dim)``.
    args : argparse.Namespace
        Training configuration (``n_critic``).
    device : torch.device
        Target compute device.

    Returns
    -------
    loss_D : float
        Mean critic loss over *n_critic* updates.
    loss_G : float
        Generator loss for this step.
    """
    B = real.size(0)
    loss_D_val = 0.0

    for _ in range(args.n_critic):
        z    = G.sample_z(B, device)
        with torch.no_grad():
            fake = G(z, cond)

        loss_D = D(fake.detach(), cond).mean() - D(real, cond).mean()

        opt_D.zero_grad(set_to_none=True)
        loss_D.backward()
        opt_D.step()

        # Weight clipping enforces Lipschitz constraint
        for p in D.parameters():
            p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

        loss_D_val += loss_D.item()

    z      = G.sample_z(B, device)
    fake   = G(z, cond)
    loss_G = -D(fake, cond).mean()

    opt_G.zero_grad(set_to_none=True)
    loss_G.backward()
    opt_G.step()

    return loss_D_val / args.n_critic, loss_G.item()


def train_step_cgan(G, D, opt_G, opt_D, real, cond, args, device):
    """
    Single training step for the Conditional GAN (BCE loss) variant.

    Uses a single critic update followed by a single generator update,
    both with Binary Cross-Entropy loss.

    Parameters
    ----------
    G : Generator
        Generator network.
    D : Critic
        Discriminator network (produces logits interpreted via BCE).
    opt_G : torch.optim.Optimizer
        Generator optimiser.
    opt_D : torch.optim.Optimizer
        Discriminator optimiser.
    real : torch.Tensor
        Real spectra batch of shape ``(B, 1, H, W)``.
    cond : torch.Tensor
        Condition vectors of shape ``(B, cond_dim)``.
    args : argparse.Namespace
        Training configuration (unused for cGAN beyond ``batch_size``).
    device : torch.device
        Target compute device.

    Returns
    -------
    loss_D : float
        Discriminator loss.
    loss_G : float
        Generator loss.
    """
    B       = real.size(0)
    ones    = torch.ones (B, device=device)
    zeros   = torch.zeros(B, device=device)
    bce     = nn.BCEWithLogitsLoss()

    # Critic step (single step — standard GAN practice)
    z    = G.sample_z(B, device)
    with torch.no_grad():
        fake = G(z, cond)

    loss_D = (bce(D(real, cond), ones) + bce(D(fake.detach(), cond), zeros)) * 0.5

    opt_D.zero_grad(set_to_none=True)
    loss_D.backward()
    opt_D.step()

    # Generator step
    z      = G.sample_z(B, device)
    fake   = G(z, cond)
    loss_G = bce(D(fake, cond), ones)

    opt_G.zero_grad(set_to_none=True)
    loss_G.backward()
    opt_G.step()

    return loss_D.item(), loss_G.item()


def train_step_wgan_gp(G, D, opt_G, opt_D, real, cond, args, device):
    """
    Single training step for the Unconditional WGAN-GP variant.

    Identical to :func:`train_step_cwgan_gp` except the condition vector is
    never passed to the generator or critic.

    Parameters
    ----------
    G : Generator
        Generator network (``cond_dim=0``).
    D : Critic
        Critic network (``cond_dim=0``).
    opt_G : torch.optim.Optimizer
        Generator optimiser.
    opt_D : torch.optim.Optimizer
        Critic optimiser.
    real : torch.Tensor
        Real spectra batch of shape ``(B, 1, H, W)``.
    cond : torch.Tensor
        Condition vectors — passed in but **not used** by this variant.
    args : argparse.Namespace
        Training configuration (``n_critic``, ``gp_lambda``).
    device : torch.device
        Target compute device.

    Returns
    -------
    loss_D : float
        Mean critic loss over *n_critic* updates.
    loss_G : float
        Generator loss for this step.
    """
    B = real.size(0)
    loss_D_val = 0.0

    for _ in range(args.n_critic):
        z    = G.sample_z(B, device)
        with torch.no_grad():
            fake = G(z)   # no cond

        gp     = gradient_penalty(D, real, fake.detach(), cond=None)
        loss_D = D(fake.detach()).mean() - D(real).mean() \
                 + args.gp_lambda * gp

        opt_D.zero_grad(set_to_none=True)
        loss_D.backward()
        opt_D.step()
        loss_D_val += loss_D.item()

    z      = G.sample_z(B, device)
    fake   = G(z)
    loss_G = -D(fake).mean()

    opt_G.zero_grad(set_to_none=True)
    loss_G.backward()
    opt_G.step()

    return loss_D_val / args.n_critic, loss_G.item()


TRAIN_STEP = {
    "cwgan_gp": train_step_cwgan_gp,
    "cwgan":    train_step_cwgan,
    "cgan":     train_step_cgan,
    "wgan_gp":  train_step_wgan_gp,
}
"""Dispatch table mapping model variant names to their training step functions."""

# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    """
    Execute the full training loop for the selected GAN variant.

    Loads the dataset, constructs the generator and critic, runs the training
    loop for ``args.epochs`` epochs, saves periodic checkpoints, and exports
    the final generator weights.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.  Expected attributes:

        - ``model`` (str) — GAN variant name, one of :data:`MODELS`.
        - ``cache`` (str) — path to the HDF5 training cache.
        - ``val_batch`` (int) — batch number held out for validation.
        - ``epochs`` (int) — total number of training epochs.
        - ``batch_size`` (int) — mini-batch size.
        - ``z_dim`` (int) — generator noise dimensionality.
        - ``lr`` (float) — base learning rate for Adam.
        - ``n_critic`` (int) — critic update steps per generator step.
        - ``gp_lambda`` (float) — gradient penalty coefficient.
        - ``out_dir`` (str) — root directory for checkpoints.
        - ``save_every`` (int) — checkpoint save interval in epochs.
        - ``resume`` (str or None) — path to a checkpoint to resume from.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model    : {args.model}")
    print(f"Device   : {device}")

    with h5py.File(args.cache, "r") as hf:
        img_h = int(hf.attrs["height"])
        img_w = int(hf.attrs["width"])
    print(f"Spectrum : ({img_h}, {img_w})")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = IMSDataset(args.cache, split="train", val_batch=args.val_batch)
    val_ds   = IMSDataset(args.cache, split="val",   val_batch=args.val_batch)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0,
        pin_memory=True, drop_last=True,
    )
    print(f"Train    : {len(train_ds)} samples  |  Val : {len(val_ds)} samples")

    # ── Models ────────────────────────────────────────────────────────────────
    # wgan_gp is unconditional → cond_dim = 0
    cond_dim_G = 0 if args.model == "wgan_gp" else 5
    cond_dim_D = 0 if args.model == "wgan_gp" else 5

    G = Generator(z_dim=args.z_dim, cond_dim=cond_dim_G,
                  img_h=img_h, img_w=img_w).to(device)
    D = Critic(cond_dim=cond_dim_D).to(device)

    lr_G = args.lr
    # cgan uses a single critic step so a higher D lr is needed
    lr_D = args.lr if args.model != "cgan" else args.lr * 2

    opt_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(0.0, 0.99))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.0, 0.99))

    print(f"G params : {sum(p.numel() for p in G.parameters()):,}")
    print(f"D params : {sum(p.numel() for p in D.parameters()):,}")

    # ── Resume ────────────────────────────────────────────────────────────────
    out_dir     = os.path.join(args.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    start_epoch = 1

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt["G_state"])
        D.load_state_dict(ckpt["D_state"])
        opt_G.load_state_dict(ckpt["opt_G_state"])
        opt_D.load_state_dict(ckpt["opt_D_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    step_fn = TRAIN_STEP[args.model]

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        G.train(); D.train()
        loss_D_ep = loss_G_ep = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:04d}", leave=False):
            x_real = batch["x"].to(device)
            cond   = batch["cond"].to(device)

            ld, lg = step_fn(G, D, opt_G, opt_D, x_real, cond, args, device)
            loss_D_ep += ld
            loss_G_ep += lg

        n = len(train_loader)
        print(
            f"Epoch {epoch:04d}/{args.epochs} | "
            f"loss_D : {loss_D_ep/n:+.4f} | "
            f"loss_G : {loss_G_ep/n:+.4f}"
        )

        if epoch % args.save_every == 0:
            path = os.path.join(out_dir, f"checkpoint_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch, "model": args.model,
                "img_h": img_h, "img_w": img_w,
                "G_state": G.state_dict(), "D_state": D.state_dict(),
                "opt_G_state": opt_G.state_dict(), "opt_D_state": opt_D.state_dict(),
            }, path)
            print(f"  → Saved {path}")

    # ── Final export ──────────────────────────────────────────────────────────
    final_path = os.path.join(out_dir, "generator_final.pt")
    torch.save({
        "model": args.model,
        "img_h": img_h, "img_w": img_w,
        "cond_dim": cond_dim_G,
        "z_dim": args.z_dim,
        "G_state": G.state_dict(),
    }, final_path)
    print(f"\nDone. Generator saved → {final_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GAN ablation variant on GC-IMS spectra.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    parser.add_argument("--model",  type=str, required=True, choices=MODELS,
                        help="GAN variant to train")
    # Data
    parser.add_argument("--cache",     type=str,   default="ims_cache.h5")
    parser.add_argument("--val_batch", type=int,   default=4,
                        help="Batch number held out as validation set")
    # Training
    parser.add_argument("--epochs",     type=int,   default=1500)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--z_dim",      type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--n_critic",   type=int,   default=5,
                        help="Critic steps per generator step (WGAN variants only)")
    parser.add_argument("--gp_lambda",  type=float, default=10.0,
                        help="Gradient penalty weight (cwgan_gp / wgan_gp only)")
    # Output
    parser.add_argument("--out_dir",    type=str,   default="checkpoints")
    parser.add_argument("--save_every", type=int,   default=500)
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()
    train(args)
