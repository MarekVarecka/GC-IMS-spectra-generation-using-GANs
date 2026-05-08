# GC-IMS Spectra Generation using GANs

> Synthetic generation of Gas Chromatography – Ion Mobility Spectrometry (GC-IMS)
> spectra for fermentation monitoring using conditional and unconditional Generative
> Adversarial Networks.

---

## Overview

This project explores four GAN architectures for generating realistic synthetic
GC-IMS spectra of microbial fermentation cultures. The goal is to augment limited
real measurement data and evaluate whether synthetic spectra are sufficient for
training downstream classifiers.

Supported organisms: *L. bulgaricus* (lb), *E. coli* (ec), *S. cerevisiae* (sc),
*P. fluorescens* (pf) — both pure and mixed two-organism cultures.

---

## GAN Variants

| Variant | Loss | Conditioning | Description |
|---|---|---|---|
| `cwgan_gp` | Wasserstein + GP | Yes | Full model — primary variant |
| `cwgan` | Wasserstein + clipping | Yes | No gradient penalty |
| `cgan` | BCE | Yes | Conditional standard GAN |
| `wgan_gp` | Wasserstein + GP | No | Unconditional baseline |

---

## Project Structure

```
GC-IMS-spectra-generation-using-GANs/
├── meaLoader.py           # Load raw .mea files from GC-IMS instrument
├── GANpreprocess.py       # Preprocessing pipeline → HDF5 cache
├── IMS_dataset.py         # PyTorch Dataset for HDF5 cache
├── IMS_models.py          # Generator and Critic architectures
├── IMS_train_models.py    # Training loops for all four variants
├── Z_make_synthetic.py    # Generate synthetic spectra from checkpoint
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/MarekVarecka/GC-IMS-spectra-generation-using-GANs.git
cd GC-IMS-spectra-generation-using-GANs

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### 1. Preprocess raw `.mea` files → HDF5 cache

```bash
python GANpreprocess.py --input data/raw/ --output ims_cache.h5
```

### 2. Train a GAN variant

```bash
# Train the full conditional WGAN-GP
python IMS_train_models.py --model cwgan_gp --cache ims_cache.h5

# Train all four variants for ablation study
python IMS_train_models.py --model cwgan_gp --cache ims_cache.h5
python IMS_train_models.py --model cwgan    --cache ims_cache.h5
python IMS_train_models.py --model cgan     --cache ims_cache.h5
python IMS_train_models.py --model wgan_gp  --cache ims_cache.h5
```

### 3. Generate synthetic spectra

```bash
# Pure cultures only
python Z_make_synthetic.py --checkpoint checkpoints/cwgan_gp_final.pt --model cwgan_gp

# Include mixed two-organism cultures
python Z_make_synthetic.py --checkpoint checkpoints/cwgan_gp_final.pt --model cwgan_gp --mixed

# Custom output path and sample count
python Z_make_synthetic.py \
    --checkpoint checkpoints/cwgan_gp_final.pt \
    --model cwgan_gp --output synthetic_cwgan_gp.h5 --n-per-cond 50 --mixed
```

---

## Data

Raw GC-IMS measurements are stored as `.mea` files. The preprocessing pipeline performs:

1. Baseline correction
2. RIP normalisation
3. Wavelet compression
4. Normalisation to `[0, 1]`
5. Export to HDF5 cache (`ims_cache.h5`)

> Raw `.mea` files are not included in this repository due to data size.

You can download them using:
```
wget https://data.mendeley.com/public-api/zip/v9gxkpdp3c/download/1
```

---

## Condition Vector

Each sample is conditioned on a 5-dimensional vector:

```
[lb, ec, sc, pf, time_norm]
 ↑── multi-hot organism flags ──↑   ↑── normalised fermentation hour
```

---

## Requirements

- Python 3.11+
- PyTorch ≥ 2.0 (CUDA recommended)
- See `requirements.txt` for full list

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{varecka2026gcims,
  author  = {Marek Varečka},
  title   = {GC-IMS Spectra Generation using Generative Adversarial Networks},
  school  = {[Your University]},
  year    = {2026}
}
```

---

## License

See [LICENSE](LICENSE) for details.
