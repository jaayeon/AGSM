<h2 align="center">
  [ICML 2026 (Spotlight)] <em>Alignment-Guided Score Matching <br>
    for Text-to-Image Alignment in Diffusion Models</em>
</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2605.30038"><img src="https://img.shields.io/badge/arXiv-2605.30038-b31b1b.svg" alt="Paper PDF"></a>
  <a href="https://jaayeon.github.io/AGSM/"><img src="https://img.shields.io/badge/Project_Page-AGSM-green" alt="Project Page"></a>
  <a href="https://huggingface.co/jaayeon/AGSM"><img src="https://img.shields.io/badge/Hugging%20Face-AGSM-yellow" alt="Hugging Face"></a>
</p>

<p align="center">
  <img width="6150" height="1686" alt="main" src="https://github.com/user-attachments/assets/2b023df2-dacb-4463-aa51-242046c1ccca" />
</p>

***TL;DR:*** AGSM is a lightweight reward-free post-training method that improves text-image alignment in diffusion models.

> [Jaa-Yeon Lee*](https://scholar.google.com/citations?user=Lw7R3sMAAAAJ&hl=ko), [Yeobin Hong*](https://yeobinhong.github.io/), [Taesung Kwon](https://star-kwon.github.io/), [Jong Chul Ye†](https://bispl.weebly.com/professor.html).
>
> KAIST

## ✨ Highlights

- **No external reward.** No reward model, preference labels, or human feedback.
- **No rollout or $x_0$ approximation.** No full denoising trajectories or Tweedie-style reconstruction.
- **No credit assignment.** Alignment signals are computed directly at each timestep.
- **Timestep-wise implicit reward.** Derived from the model's own score-matching likelihood via Plackett-Luce modeling.
- **Alignment-guided score matching.** Turns text-image alignment into a score-matching target on the forward noising process.
- **Positive/negative soft tokens.** Lightweight guidance for matched and mismatched text-image pairs.

## 🔥 News
- [2026.06.21] Our code is released on Github.
- [2026.05.29] Our paper is released on arXiv.
- [2026.05.01] Our paper is accepted to ICML 2026 as a Spotlight.

## Setup

```bash
git clone git@github.com:jaayeon/AGSM.git
cd AGSM
conda env create -f environment.yaml
conda activate agsm
```

If you already have a compatible Python environment:

```bash
pip install -r requirements.txt
```

## Repository Layout

```text
train.py              # final DDP AGSM training code
sample.py             # generate samples from learned AGSM tokens
eval.py               # ImageReward/CLIP/PickScore evaluation
sampler.py            # SD1.5, SDXL, and SD3 samplers with AGSM tokens
transformer.py        # token-injected diffusion model wrappers
dataset/              # COCO dataset loader
scripts/              # clean launch scripts
checkpoints/          # released AGSM token checkpoints
```

## Data

Set `DATADIR` to the parent directory containing COCO:

```text
DATADIR/
└── coco/
    ├── train2017/
    ├── val2017/
    └── annotations/
        ├── captions_train2017.json
        └── captions_val2017.json
```

## Training

```bash
DATADIR=/path/to/datasets \
MODEL=sd3 \
NPROC_PER_NODE=1 \
scripts/train_coco.sh
```

The trainer loads images directly by default and writes checkpoints/validation samples under `LOGDIR`, which defaults to `./outputs/train`.
If you already have precomputed latents and prompt embeddings, set `USE_PRECOMPUTED_ENCODINGS=true` and `ENCODING_DIR=/path/to/encodings_sd3`.

## Script Configuration

The launch scripts are configured through environment variables. Common overrides are:

| Variable | Used by | Description |
| --- | --- | --- |
| `MODEL` | train/sample/eval | One of `sd3`, `sd1.5`, `sdxl`. |
| `DATADIR` | train/sample/eval | Parent directory containing `coco/`. |
| `NPROC_PER_NODE` | train | Number of DDP processes/GPUs for `torchrun`. |
| `LOGDIR` | train | Training output directory. |
| `NUM_SAMPLES` | sample/eval | `-1` for all COCO validation images, or an integer. |
| `OUTPUT_DIR` | sample | Sampling output directory. |
| `RESULT_ROOT` | eval | Directory containing sampled images/results JSON. |
| `BENCHMARKS` | eval | Comma-separated evaluation metrics. |

## Sampling

Released AGSM token checkpoints are included under `checkpoints/` and mirrored on [Hugging Face](https://huggingface.co/jaayeon/AGSM). To sample with the default SD3 checkpoint:

| Model | Checkpoint | CFG |
| --- | --- | ---: |
| `sd3` | `checkpoints/sd3` | 4.0 |
| `sd1.5` | `checkpoints/sd1.5` | 7.0 |
| `sdxl` | `checkpoints/sdxl` | 7.0 |

```bash
DATADIR=/path/to/datasets \
MODEL=sd3 \
scripts/sample_coco.sh
```

Use `MODEL=sd1.5` or `MODEL=sdxl` to use the corresponding released checkpoint. You can still override `CHECKPOINT_DIR`, `CFG_SCALE`, or `UNCON2NEG`.

This creates an image folder and a matching `results-*.json` file under `OUTPUT_DIR`.

## Evaluation

After sampling, evaluate the matching results:

```bash
DATADIR=/path/to/datasets \
MODEL=sd3 \
scripts/eval_coco.sh
```

By default, samples are written to `./outputs/samples/${MODEL}` and evaluation reads from the same location.

Use `BENCHMARKS` to choose metrics, for example:

```bash
BENCHMARKS=ImageReward-v1.0,CLIP,PickScore scripts/eval_coco.sh
```

## Notes

- `WANDB=false` by default. Set `WANDB=true` in `scripts/train_coco.sh` if you want W&B logging.
- Generated data, logs, checkpoints, and local datasets are ignored by git.

## Citation

```bibtex
@article{lee2026alignment,
  title={Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models},
  author={Lee, Jaa-Yeon and Hong, Yeobin and Kwon, Taesung and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2605.30038},
  year={2026}
}
```
