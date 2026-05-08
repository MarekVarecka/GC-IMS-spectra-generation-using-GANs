"""
IMS_models.py
=============
All four GAN variants for ablation study.

Models
------
  cwgan_gp   Conditional WGAN + Gradient Penalty           (full model)
  cwgan      Conditional WGAN + weight clipping            (no GP)
  cgan       Conditional GAN  + BCE loss                   (no Wasserstein)
  wgan_gp    Unconditional WGAN + Gradient Penalty         (no conditioning)

Shared components
-----------------
  ConditionEmbedding   - condition → dense embedding (used by conditional models)
  GenBlock             - AdaIN upsampling block
  Generator            - shared across all four models
  CriticBlock          - spectral-norm downsampling block
  Critic               - conditional (cond_dim > 0) or unconditional (cond_dim = 0)
  gradient_penalty     - WGAN-GP penalty (used by cwgan_gp and wgan_gp)

References
----------
  Goodfellow et al.  (2014) - Generative Adversarial Nets
  Arjovsky   et al.  (2017) - Wasserstein GAN
  Gulrajani  et al.  (2017) - Improved Training of Wasserstein GANs
  Mirza & Osindero   (2014) - Conditional GAN
  Miyato     et al.  (2018) - Spectral Normalisation for GANs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ── Condition embedding ───────────────────────────────────────────────────────

class ConditionEmbedding(nn.Module):
    """
    Projects a raw condition vector into a dense embedding space.

    The condition vector encodes biological metadata:
      - Indices 0-3 : multi-hot organism flags  [lb, ec, sc, pf]
      - Index    4  : normalised fermentation hour  (hour / MAX_HOURS)

    Parameters
    ----------
    cond_dim  : int  Dimensionality of the raw condition vector.  Default: 5.
    embed_dim : int  Dimensionality of the output embedding.      Default: 64.

    Input  : (B, cond_dim)
    Output : (B, embed_dim)
    """

    def __init__(self, cond_dim: int = 5, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


# ── Generator blocks ──────────────────────────────────────────────────────────

class GenBlock(nn.Module):
    """
    Conditional upsampling block (AdaIN modulation).
    Doubles spatial resolution via ConvTranspose2d.

    Input  : x (B, in_ch, H, W),  embed (B, embed_dim)
    Output : x (B, out_ch, H*2, W*2)
    """

    def __init__(self, in_ch: int, out_ch: int, embed_dim: int):
        super().__init__()
        self.conv  = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn    = nn.BatchNorm2d(out_ch, affine=False)
        self.scale = nn.Linear(embed_dim, out_ch)
        self.shift = nn.Linear(embed_dim, out_ch)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        x     = self.bn(self.conv(x))
        scale = self.scale(embed).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift(embed).unsqueeze(-1).unsqueeze(-1)
        return F.leaky_relu(x * (1.0 + scale) + shift, 0.2, inplace=True)


class GenBlockUncond(nn.Module):
    """
    Unconditional upsampling block (plain BN, no AdaIN).
    Used by wgan_gp which has no condition vector.

    Input  : x (B, in_ch, H, W)
    Output : x (B, out_ch, H*2, W*2)
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.bn(self.conv(x)), 0.2, inplace=True)


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Shared generator backbone for all four model variants.

    When cond_dim > 0 the generator uses AdaIN blocks (conditional).
    When cond_dim = 0 it falls back to unconditional BN blocks.

    Spatial progression (img_h=690, img_w=128):
        Seed      :  86 ×  16
        Block 1   : 172 ×  32
        Block 2   : 344 ×  64
        Block 3   : 688 × 128
        Resize    : 690 × 128  ← exact target

    Parameters
    ----------
    z_dim     : int   Noise dimensionality.              Default: 128.
    cond_dim  : int   Condition dimensionality (0 = uncond). Default: 5.
    base_ch   : int   Seed channel width.                Default: 256.
    embed_dim : int   Condition embedding width.         Default: 64.
    img_h     : int   Target height.                     Default: 690.
    img_w     : int   Target width.                      Default: 128.

    Input  : z (B, z_dim),  cond (B, cond_dim)  [cond ignored when cond_dim=0]
    Output : x (B, 1, img_h, img_w)  values in [0, 1]
    """

    def __init__(self, z_dim: int = 128, cond_dim: int = 5,
                 base_ch: int = 256, embed_dim: int = 64,
                 img_h: int = 690, img_w: int = 128):
        super().__init__()
        self.z_dim    = z_dim
        self.cond_dim = cond_dim
        self.base_ch  = base_ch
        self.img_h    = img_h
        self.img_w    = img_w
        self.init_h   = max(1, img_h // 8)
        self.init_w   = max(1, img_w // 8)

        self.conditional = cond_dim > 0

        if self.conditional:
            self.cond_embed = ConditionEmbedding(cond_dim, embed_dim)
            fc_in = z_dim + embed_dim
        else:
            fc_in = z_dim

        self.fc      = nn.Linear(fc_in, base_ch * self.init_h * self.init_w)
        self.bn_init = nn.BatchNorm2d(base_ch)

        if self.conditional:
            self.block1 = GenBlock(base_ch, 128, embed_dim)
            self.block2 = GenBlock(128,      64, embed_dim)
            self.block3 = GenBlock( 64,      32, embed_dim)
        else:
            self.block1 = GenBlockUncond(base_ch, 128)
            self.block2 = GenBlockUncond(128,      64)
            self.block3 = GenBlockUncond( 64,      32)

        self.out = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        if self.conditional:
            embed = self.cond_embed(cond)
            inp   = torch.cat([z, embed], dim=1)
        else:
            embed = None
            inp   = z

        x = self.fc(inp).view(-1, self.base_ch, self.init_h, self.init_w)
        x = F.leaky_relu(self.bn_init(x), 0.2, inplace=True)

        if self.conditional:
            x = self.block1(x, embed)
            x = self.block2(x, embed)
            x = self.block3(x, embed)
        else:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)

        x = torch.sigmoid(self.out(x))

        if x.shape[2] != self.img_h or x.shape[3] != self.img_w:
            x = F.interpolate(x, size=(self.img_h, self.img_w),
                              mode="bilinear", align_corners=False)
        return x

    def sample_z(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.randn(n, self.z_dim, device=device)


# ── Critic / Discriminator blocks ─────────────────────────────────────────────

class CriticBlock(nn.Module):
    """
    Strided downsampling block with spectral norm + instance norm.
    Halves spatial resolution.

    Input  : (B, in_ch,  H, W)
    Output : (B, out_ch, H//2, W//2)
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        )
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.norm(self.conv(x)), 0.2, inplace=True)


# ── Critic ────────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """
    Shared critic/discriminator for all four variants.

    When cond_dim > 0 the condition is spatially broadcast and concatenated
    to the input feature map (conditional discrimination).
    When cond_dim = 0 only the spectrum is fed in (unconditional).

    For cgan the same architecture is used but the final head outputs a
    logit interpreted with BCEWithLogitsLoss instead of a Wasserstein score.

    Architecture
    ------------
    Input  : (B, 1 + cond_dim, H,    W   )
    Block1 : (B, base_ch,      H/2,  W/2 )
    Block2 : (B, base_ch×2,   H/4,  W/4 )
    Block3 : (B, base_ch×4,   H/8,  W/8 )
    Block4 : (B, base_ch×8,   H/16, W/16)
    Pool   : (B, base_ch×8,   4,    4   )
    FC     : (B, 1)

    Parameters
    ----------
    cond_dim : int  Condition dimensionality (0 = unconditional). Default: 5.
    base_ch  : int  Base channel width.                          Default: 32.

    Input  : x (B, 1, H, W),  cond (B, cond_dim)
    Output : score (B,)   — unbounded for WGAN variants, logit for cgan
    """

    def __init__(self, cond_dim: int = 5, base_ch: int = 32):
        super().__init__()
        in_ch = 1 + cond_dim

        self.blocks = nn.Sequential(
            CriticBlock(in_ch,       base_ch),
            CriticBlock(base_ch,     base_ch * 2),
            CriticBlock(base_ch * 2, base_ch * 4),
            CriticBlock(base_ch * 4, base_ch * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc   = spectral_norm(nn.Linear(base_ch * 8 * 4 * 4, 1))

    def forward(self, x: torch.Tensor,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        B, C, H, W = x.shape
        if cond is not None:
            cond_map = cond.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
            x = torch.cat([x, cond_map], dim=1)
        x = self.blocks(x)
        x = self.pool(x).view(B, -1)
        return self.fc(x).squeeze(1)


# ── Gradient penalty ──────────────────────────────────────────────────────────

def gradient_penalty(critic: Critic, real: torch.Tensor,
                     fake: torch.Tensor,
                     cond: torch.Tensor | None = None) -> torch.Tensor:
    """
    WGAN-GP gradient penalty (Gulrajani et al., 2017).

    GP = E[ (‖∇_{x̂} D(x̂)‖₂ − 1)² ]
    where x̂ = α·x_real + (1−α)·x_fake,  α ~ U(0,1)

    Parameters
    ----------
    critic : Critic
    real   : (B, 1, H, W)  real spectra
    fake   : (B, 1, H, W)  generated spectra (detached)
    cond   : (B, cond_dim) or None for unconditional

    Returns
    -------
    gp : scalar tensor
    """
    B     = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=real.device)
    interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)

    score = critic(interp, cond)

    grads = torch.autograd.grad(
        outputs=score, inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True,
    )[0]

    return ((grads.view(B, -1).norm(2, dim=1) - 1.0) ** 2).mean()
