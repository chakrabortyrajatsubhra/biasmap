# BiasMap

**Leveraging Cross-Attentions to Discover and Mitigate Hidden Social Biases in Text-to-Image Generation**

> Accepted to **ACM SIGKDD 2026**.
> [Paper (DOI)](https://doi.org/10.48550/arXiv.2509.13496) &nbsp;·&nbsp; [Project page](https://chakrabortyrajatsubhra.github.io/biasmap/)

BiasMap uncovers latent concept-level representational bias in U-Net-based Stable Diffusion models using cross-attention attribution maps, and mitigates it through energy-guided diffusion sampling that minimizes the spatial overlap (SoftIoU) between a demographic concept and a profession concept during denoising.

This repository contains the mitigation code used in the paper, in two entry points:

| Script | Demographic axis | Default model |
| --- | --- | --- |
| `src/run_gender.py` | Gender (male ↔ female) | `runwayml/stable-diffusion-v1-5` |
| `src/run_race.py` | Race (white ↔ African-American) | `runwayml/stable-diffusion-v1-5` (SD 2.0 optional) |

---

## 1. Requirements

- Linux with an NVIDIA GPU (experiments in the paper used a single **A100**)
- **CUDA-enabled PyTorch**
- Python 3.9–3.11
- A Hugging Face account (the SD checkpoints are gated on first download)

---

## 2. Setup

BiasMap builds on two external repositories: **Fair Diffusion** (the semantic-guidance pipeline we extend) and **OVAM** (Open-Vocabulary Attention Maps, used to extract the cross-attention attribution maps). The scripts import `ovam` directly, so both must live next to the BiasMap code in the same working folder.

### 2.1 Clone this repository

```bash
git clone https://github.com/chakrabortyrajatsubhra/biasmap.git
cd biasmap
```

### 2.2 Clone the two dependencies into the same folder

From inside the `biasmap/` folder:

```bash
# Fair Diffusion (semantic guidance pipeline we extend)
git clone https://github.com/ml-research/semantic-image-editing.git fair-diffusion

# OVAM (open-vocabulary cross-attention maps)
git clone https://github.com/vpulab/ovam.git
```

After this step your folder should look like:

```
biasmap/
├── src/
│   ├── run_gender.py
│   └── run_race.py
├── fair-diffusion/        # cloned
├── ovam/                  # cloned  ← provides the `ovam` package imported by both scripts
├── requirements.txt
├── LICENSE
└── README.md
```

> **Why the same folder?** Both scripts do `from ovam import StableDiffusionHooker`. Cloning `ovam` into the project root (or installing it, see below) puts the `ovam` package on the import path. If you prefer, you can `pip install -e ./ovam` instead of relying on the local folder.

### 2.3 Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate

# Install a CUDA build of PyTorch first (match your CUDA version):
# see https://pytorch.org/get-started/locally/ for the exact command, e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then the rest:
pip install -r requirements.txt

# Make the cloned OVAM package importable (recommended):
pip install -e ./ovam
```

### 2.4 Authenticate with Hugging Face (first run only)

```bash
huggingface-cli login
```

Accept the model license for `runwayml/stable-diffusion-v1-5` (and `stabilityai/stable-diffusion-2-base` if you use the SD 2.0 option) on the Hugging Face model pages before running.

---

## 3. Running BiasMap

All commands are run from the repository root with the environment activated.

### 3.1 Gender debiasing (full benchmark)

Runs energy-guided mitigation over all 20 professions and 100 seeds each, and writes mitigated images plus a CSV of per-image IoU reductions:

```bash
python src/run_gender.py
```

Outputs are written to `Gender_outputs/`, including:
- `mitigated_<profession>_seed<seed>.png` for each generation
- `mitigated_ious.csv` with columns: `profession, seed, editing_prompt, mitigated_iou, original_iou, iou_reduction, area_reduction, image_path`

### 3.2 Race debiasing (full benchmark)

```bash
python src/run_race.py
```

Outputs are written to `Race_outputs/` with the same CSV schema.

### 3.3 Single qualitative examples (race script)

`src/run_race.py` also reproduces the qualitative figures from the paper via named modes. The optional second argument selects the model (`sd15` default, or `sd2` for SD 2.0):

```bash
python src/run_race.py architect          # architect, SD 1.5, fixed seed
python src/run_race.py musician            # musician, seeds 36/72/87/88
python src/run_race.py musician sd2        # same, on SD 2.0
python src/run_race.py pilot               # pilot, seeds 36/72/87/88
python src/run_race.py doctor              # doctor, race attribution
python src/run_race.py mechanic            # mechanic, race, kitchen + office scenes
```

Each mode saves its images under a dedicated `*_example/seed_<seed>/` folder.

---

## 4. Key hyperparameters

Defaults match the paper. You can edit them in the `batch_run_ovam_bias_mitigation(...)` call or function defaults:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `guidance_scale` (γ) | 7.0–7.5 | classifier-free guidance strength |
| `regressor_scale` (λ) | 100.0 | energy-guidance scale (higher = stronger debiasing, lower fidelity) |
| `threshold_percentile` (q) | 70 | percentile for high-attention masks |
| `num_inference_steps` | 50 | denoising steps |
| `seeds` | `range(100)` | seeds per profession |

The λ sweep in the paper (Tables 7–8) shows the fairness–fidelity tradeoff; λ = 100 sits near the Pareto knee.

---

## 5. Citation

```bibtex
@inproceedings{biasmap2026,
  title     = {BiasMap: Leveraging Cross-Attentions to Discover and
               Mitigate Hidden Social Biases in Text-to-Image Generation},
  author    = {Chakraborty, Rajatsubhra and Che, Xujun and Xu, Depeng
               and Faklaris, Cori and Niu, Xi and Yuan, Shuhan},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining (KDD)},
  year      = {2026},
  doi       = {10.1145/3770855.3818098}
}
```

## 6. Acknowledgements

This work builds on [Fair Diffusion / semantic-image-editing](https://github.com/ml-research/semantic-image-editing) and [OVAM](https://github.com/vpulab/ovam). We thank their authors for releasing their code.

## License

Released under the MIT License (see [`LICENSE`](LICENSE)). Note that the cloned dependencies and the Stable Diffusion checkpoints carry their own licenses.
