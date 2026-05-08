"""
IMS_models.py
=============
PyTorch neural network definitions for all four GAN ablation variants.

Models
------
``cwgan_gp``
    Conditional WGAN + Gradient Penalty (primary / full model).
``cwgan``
    Conditional WGAN + weight clipping (no GP).
``cgan``
    Conditional GAN + BCE loss (no Wasserstein distance).
``wgan_gp``
    Unconditional WGAN + Gradient Penalty (no conditioning).

Shared components
-----------------
:class:`ConditionEmbedding`
    Maps a raw condition vector to a dense embedding used by conditional models.
:class:`GenBlock`
    Conditional AdaIN upsampling block for the generator.
:class:`GenBlockUncond`
    Unconditional upsampling block (plain BatchNorm) for ``wgan_gp``.
:class:`Generator`
    Shared generator backbone for all four variants.
:class:`CriticBlock`
    Spectral-norm downsampling block for the critic.
:class:`Critic`
    Shared critic/discriminator for all four variants.
:func:`gradient_penalty`
    WGAN-GP gradient penalty term (Gulrajani et al., 2017).

References
----------
Goodfellow et al. (2014) — Generative Adversarial Nets.
Arjovsky et al. (2017) — Wasserstein GAN.
Gulrajani et al. (2017) — Improved Training of Wasserstein GANs.
Mirza & Osindero (2014) — Conditional Generative Adversarial Nets.
Miyato et al. (2018) — Spectral Normalisation for GANs.
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

    - Indices 0–3 : multi-hot organism flags ``[lb, ec, sc, pf]``.
    - Index 4 : normalised fermentation hour ``(hour / MAX_HOURS)``.

    Parameters
    ----------
    cond_dim : int, optional
        Dimensionality of the raw condition vector. Default is ``5``.
    embed_dim : int, optional
        Dimensionality of the output embedding. Default is ``64``.

    Notes
    -----
    Input shape  : ``(B, cond_dim)``
    Output shape : ``(B, embed_dim)``
    """

    def __init__(self, cond_dim: int = 5, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cond : torch.Tensor
            Raw condition vectors of shape ``(B, cond_dim)``.

        Returns
        -------
        torch.Tensor
            Dense condition embeddings of shape ``(B, embed_dim)``.
        """
        return self.net(cond)


# ── Generator blocks ──────────────────────────────────────────────────────────

class GenBlock(nn.Module):
    """
    Conditional upsampling block with Adaptive Instance Normalisation (AdaIN).

    Doubles the spatial resolution via :class:`~torch.nn.ConvTranspose2d` and
    modulates the normalised activations with scale/shift parameters derived
    from the condition embedding.

    Parameters
    ----------
    in_ch : int
        Number of input feature-map channels.
    out_ch : int
        Number of output feature-map channels.
    embed_dim : int
        Dimensionality of the condition embedding.

    Notes
    -----
    Input  : ``x`` of shape ``(B, in_ch, H, W)``; ``embed`` of shape ``(B, embed_dim)``
    Output : ``(B, out_ch, H*2, W*2)``
    """

    def __init__(self, in_ch: int, out_ch: int, embed_dim: int):
        super().__init__()
        self.conv  = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn    = nn.BatchNorm2d(out_ch, affine=False)
        self.scale = nn.Linear(embed_dim, out_ch)
        self.shift = nn.Linear(embed_dim, out_ch)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Feature map of shape ``(B, in_ch, H, W)``.
        embed : torch.Tensor
            Condition embedding of shape ``(B, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Upsampled feature map of shape ``(B, out_ch, H*2, W*2)``.
        """
        x     = self.bn(self.conv(x))
        scale = self.scale(embed).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift(embed).unsqueeze(-1).unsqueeze(-1)
        return F.leaky_relu(x * (1.0 + scale) + shift, 0.2, inplace=True)


class GenBlockUncond(nn.Module):
    """
    Unconditional upsampling block with plain BatchNorm.

    Used exclusively by the ``wgan_gp`` variant which has no condition vector.

    Parameters
    ----------
    in_ch : int
        Number of input feature-map channels.
    out_ch : int
        Number of output feature-map channels.

    Notes
    -----
    Input  : ``(B, in_ch, H, W)``
    Output : ``(B, out_ch, H*2, W*2)``
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Feature map of shape ``(B, in_ch, H, W)``.

        Returns
        -------
        torch.Tensor
            Upsampled feature map of shape ``(B, out_ch, H*2, W*2)``.
        """
        return F.leaky_relu(self.bn(self.conv(x)), 0.2, inplace=True)


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Shared generator backbone for all four GAN ablation variants.

    When ``cond_dim > 0`` the generator uses AdaIN blocks
    (:class:`GenBlock`) for conditional generation.
    When ``cond_dim = 0`` it falls back to unconditional BatchNorm blocks
    (:class:`GenBlockUncond`).

    Spatial resolution progression (default ``img_h=690``, ``img_w=128``):

    .. code-block:: text

        Seed    :  86 ×  16
        Block 1 : 172 ×  32
        Block 2 : 344 ×  64
        Block 3 : 688 × 128
        Resize  : 690 × 128  ← exact target via bilinear interpolation

    Parameters
    ----------
    z_dim : int, optional
        Noise vector dimensionality. Default is ``128``.
    cond_dim : int, optional
        Condition vector dimensionality.  Set to ``0`` for unconditional
        generation (``wgan_gp``). Default is ``5``.
    base_ch : int, optional
        Number of channels in the initial seed feature map. Default is ``256``.
    embed_dim : int, optional
        Condition embedding width (used only when ``cond_dim > 0``).
        Default is ``64``.
    img_h : int, optional
        Target output height in pixels. Default is ``690``.
    img_w : int, optional
        Target output width in pixels. Default is ``128``.

    Notes
    -----
    Input  : ``z`` of shape ``(B, z_dim)``; ``cond`` of shape ``(B, cond_dim)``
             (ignored when ``cond_dim = 0``).
    Output : ``(B, 1, img_h, img_w)`` — values in ``[0, 1]`` after sigmoid.
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
        """
        Generate a batch of synthetic GC-IMS spectra.

        Parameters
        ----------
        z : torch.Tensor
            Noise vectors of shape ``(B, z_dim)`` sampled from
            :math:`\mathcal{N}(0, I)`.
        cond : torch.Tensor or None, optional
            Condition vectors of shape ``(B, cond_dim)``.
            Ignored when ``cond_dim = 0``. Default is ``None``.

        Returns
        -------
        torch.Tensor
            Synthetic spectra of shape ``(B, 1, img_h, img_w)`` with values
            in ``[0, 1]``.
        """
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
        """
        Sample a batch of noise vectors from :math:`\mathcal{N}(0, I)`.

        Parameters
        ----------
        n : int
            Number of noise vectors to sample.
        device : torch.device
            Target device for the returned tensor.

        Returns
        -------
        torch.Tensor
            Noise tensor of shape ``(n, z_dim)``.
        """
        return torch.randn(n, self.z_dim, device=device)


# ── Critic / Discriminator blocks ─────────────────────────────────────────────

class CriticBlock(nn.Module):
    """
    Strided downsampling block with spectral normalisation and instance norm.

    Halves the spatial resolution in both dimensions.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.

    Notes
    -----
    Input  : ``(B, in_ch, H, W)``
    Output : ``(B, out_ch, H//2, W//2)``
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        )
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Feature map of shape ``(B, in_ch, H, W)``.

        Returns
        -------
        torch.Tensor
            Downsampled feature map of shape ``(B, out_ch, H//2, W//2)``.
        """
        return F.leaky_relu(self.norm(self.conv(x)), 0.2, inplace=True)


# ── Critic ────────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """
    Shared critic/discriminator backbone for all four GAN variants.

    When ``cond_dim > 0`` the condition vector is spatially broadcast and
    concatenated to the input feature map before the first convolutional
    block (conditional discrimination).
    When ``cond_dim = 0`` only the spectrum is fed in (unconditional).

    For the ``cgan`` variant the same architecture is used but the final
    scalar output is interpreted as a logit for
    :class:`~torch.nn.BCEWithLogitsLoss` rather than a Wasserstein score.

    Architecture
    ------------
    .. code-block:: text

        Input  : (B, 1 + cond_dim, H, W)
        Block 1: (B, base_ch,     H/2,  W/2)
        Block 2: (B, base_ch×2,   H/4,  W/4)
        Block 3: (B, base_ch×4,   H/8,  W/8)
        Block 4: (B, base_ch×8,   H/16, W/16)
        Pool   : (B, base_ch×8,   4,    4)
        FC     : (B, 1)

    Parameters
    ----------
    cond_dim : int, optional
        Condition vector dimensionality.  ``0`` for unconditional.
        Default is ``5``.
    base_ch : int, optional
        Base channel width for the convolutional blocks. Default is ``32``.

    Notes
    -----
    Input  : ``x`` of shape ``(B, 1, H, W)``; ``cond`` of shape ``(B, cond_dim)``
    Output : unbounded scalar score per sample ``(B,)`` for WGAN variants;
             logit for ``cgan``.
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
        """
        Compute critic scores for a batch of spectra.

        Parameters
        ----------
        x : torch.Tensor
            Spectra of shape ``(B, 1, H, W)``.
        cond : torch.Tensor or None, optional
            Condition vectors of shape ``(B, cond_dim)``.
            Pass ``None`` for unconditional operation. Default is ``None``.

        Returns
        -------
        torch.Tensor
            Per-sample scores of shape ``(B,)``.
        """
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
    Compute the WGAN-GP gradient penalty (Gulrajani et al., 2017).

    The penalty is defined as:

    .. math::

        \mathrm{GP} = \mathbb{E}\left[\left(\|\nabla_{\hat{x}}
        D(\hat{x})\|_2 - 1\right)^2\right]

    where :math:`\hat{x} = \alpha x_{\mathrm{real}} + (1-\alpha)
    x_{\mathrm{fake}}` with :math:`\alpha \sim \mathcal{U}(0, 1)`.

    Parameters
    ----------
    critic : Critic
        The critic network :math:`D`.
    real : torch.Tensor
        Real spectra of shape ``(B, 1, H, W)``.
    fake : torch.Tensor
        Generated spectra of shape ``(B, 1, H, W)``. Should be detached
        from the generator graph before passing.
    cond : torch.Tensor or None, optional
        Condition vectors of shape ``(B, cond_dim)``, or ``None`` for
        unconditional critics. Default is ``None``.

    Returns
    -------
    torch.Tensor
        Scalar gradient penalty term.
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
