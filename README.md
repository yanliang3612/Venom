# Venom

Educational PyTorch implementations of the main discrete-time and continuous-time diffusion families:

- **DDPM**: linear beta schedule, epsilon prediction, fixed posterior variance.
- **Improved DDPM**: cosine schedule and learned-range variance.
- **ADM / Guided Diffusion**: class-conditional U-Net with attention and optional classifier guidance.
- **Classifier-Free Guidance**: conditional dropout during training and guidance-scale sampling.
- **EDM**: Karras preconditioning, log-normal noise training, Euler/Heun sampler.
- **DPM-Solver / DPM-Solver++**: lightweight first-order fast samplers for DDPM checkpoints.
- **DiT**: optional diffusion transformer backbone for MNIST-sized images.
- **NCSN / NCSNv2**: denoising score matching over geometric noise scales.
- **Score SDE**: VP, VE, and sub-VP SDE objectives with predictor-corrector sampling.
- **PFGM / PFGM++**: Poisson-flow-style perturbation kernels and Karras-style sampling.
- **Rectified Flow / Flow Matching**: continuous velocity fields from noise to data.
- **Conditional Flow Matching / OT-CFM**: conditional paths and dependency-light minibatch OT pairing.
- **Stochastic Interpolants**: noisy interpolating paths that unify flow and diffusion-style dynamics.
- **Progressive Distillation**: teacher-student helper for compressing DDPM-family samplers.
- **Consistency Models**: one/few-step consistency training with EDM-style preconditioning.
- **Shortcut Models**: flow models conditioned on the requested step size.
- **MeanFlow**: average-velocity training for one-step or few-step generation.

The default dataset is MNIST so the code stays small enough to read and modify.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you prefer requirements only:

```bash
pip install -r requirements.txt
```

## Train MNIST Models

```bash
# Original DDPM
python train.py --variant ddpm --epochs 5

# Improved DDPM: cosine schedule + learned-range variance
python train.py --variant improved-ddpm --epochs 5

# ADM-style class-conditional model
python train.py --variant adm --epochs 5

# Classifier-free guidance model
python train.py --variant cfg --epochs 5 --class-dropout 0.1

# EDM objective and Karras sampler
python train.py --variant edm --epochs 5 --sample-steps 32

# NCSN / NCSNv2 score matching
python train.py --variant ncsn --epochs 5
python train.py --variant ncsnv2 --epochs 5

# Continuous-time Score SDE variants
python train.py --variant score-sde-vp --epochs 5 --sample-steps 250
python train.py --variant score-sde-ve --epochs 5 --sample-steps 250
python train.py --variant score-sde-subvp --epochs 5 --sample-steps 250

# PFGM / PFGM++
python train.py --variant pfgm --epochs 5 --sample-steps 32
python train.py --variant pfgm++ --epochs 5 --sample-steps 32

# Flow and interpolant models
python train.py --variant rectified-flow --epochs 5 --sample-steps 50
python train.py --variant flow-matching --epochs 5 --sample-steps 50
python train.py --variant conditional-flow-matching --epochs 5 --sample-steps 50
python train.py --variant ot-cfm --epochs 5 --sample-steps 50
python train.py --variant stochastic-interpolants --epochs 5 --sample-steps 50

# One-step and few-step families
python train.py --variant consistency --epochs 5 --sample-steps 1
python train.py --variant shortcut --epochs 5 --sample-steps 1
python train.py --variant meanflow --epochs 5 --sample-steps 1

# Swap the U-Net for a small DiT backbone
python train.py --variant ddpm --backbone dit --epochs 5
python train.py --variant rectified-flow --backbone dit --epochs 5
python train.py --variant meanflow --backbone dit --epochs 5
```

Progressive distillation starts from a trained DDPM-family teacher:

```bash
python -m venom.train_progressive_distill_mnist \
  --teacher-checkpoint runs/mnist/improved-ddpm/model_005.pt \
  --student-steps 50 \
  --epochs 3
```

Checkpoints and preview grids are written to `runs/mnist/<variant>/`.

## Sample

```bash
python sample.py \
  --checkpoint runs/mnist/ddpm/model_005.pt \
  --sampler ddim \
  --sample-steps 50 \
  --num-samples 64 \
  --out samples.png
```

Fast samplers for DDPM-family checkpoints:

```bash
python sample.py --checkpoint runs/mnist/improved-ddpm/model_005.pt --sampler dpm-solver --sample-steps 20
python sample.py --checkpoint runs/mnist/improved-ddpm/model_005.pt --sampler dpm-solver++ --sample-steps 20
```

Classifier-free guidance:

```bash
python sample.py \
  --checkpoint runs/mnist/cfg/model_005.pt \
  --sampler dpm-solver++ \
  --sample-steps 20 \
  --labels 0,1,2,3,4,5,6,7,8,9 \
  --guidance-scale 3.0
```

Continuous-time checkpoints use their native samplers:

```bash
python sample.py --checkpoint runs/mnist/edm/model_005.pt --sample-steps 32
python sample.py --checkpoint runs/mnist/score-sde-ve/model_005.pt --sample-steps 250
python sample.py --checkpoint runs/mnist/pfgm++/model_005.pt --sample-steps 32
python sample.py --checkpoint runs/mnist/rectified-flow/model_005.pt --sample-steps 50
python sample.py --checkpoint runs/mnist/meanflow/model_005.pt --sample-steps 1
```

## Classifier Guidance

Train a timestep-conditioned noised classifier:

```bash
python -m venom.train_classifier_mnist --epochs 3
```

Then sample a class-conditional ADM checkpoint with classifier guidance:

```bash
python sample.py \
  --checkpoint runs/mnist/adm/model_005.pt \
  --sampler ddpm \
  --labels 0,1,2,3,4,5,6,7,8,9 \
  --classifier-checkpoint runs/mnist/classifier/classifier_003.pt \
  --classifier-scale 1.0
```

## Python API

```python
import torch

from venom import GaussianDiffusion, UNet2D
from venom.samplers import DPMSolverSampler

model = UNet2D(image_channels=1, base_channels=64)
diffusion = GaussianDiffusion(model, timesteps=1000)

x = torch.randn(8, 1, 28, 28)
loss = diffusion.training_loss(x)

sampler = DPMSolverSampler(diffusion, steps=20, algorithm="dpmsolver++")
samples = sampler.sample(batch_size=8, device=x.device)
```

Continuous-time API:

```python
import torch

from venom import ScoreSDEDiffusion, UNet2D, VESDE

model = UNet2D(image_channels=1, base_channels=64)
diffusion = ScoreSDEDiffusion(model, VESDE())

x = torch.randn(8, 1, 28, 28)
loss = diffusion.training_loss(x)
samples = diffusion.sample(batch_size=8, device=x.device, steps=250)
```

Flow matching API:

```python
import torch

from venom import RectifiedFlow, UNet2D

model = UNet2D(image_channels=1, base_channels=64)
flow = RectifiedFlow(model)

x = torch.randn(8, 1, 28, 28)
loss = flow.training_loss(x)
samples = flow.sample(batch_size=8, device=x.device, steps=50)
```

One-step API:

```python
import torch

from venom import MeanFlow, UNet2D

model = UNet2D(image_channels=1, base_channels=64)
meanflow = MeanFlow(model)

x = torch.randn(8, 1, 28, 28)
loss = meanflow.training_loss(x)
samples = meanflow.sample(batch_size=8, device=x.device, steps=1)
```

Progressive distillation API:

```python
from venom import ProgressiveDistillation

distiller = ProgressiveDistillation(student_diffusion, teacher_diffusion, student_steps=50)
loss = distiller.training_loss(images)
```

## Notes

This package is intended as a clean research scaffold, not a drop-in reproduction
of the full OpenAI `guided-diffusion` or EDM codebases. The APIs separate:

- model architecture: `venom.models`
- beta/noise schedules: `venom.schedules`
- DDPM-family objective: `venom.diffusion`
- EDM objective: `venom.edm`
- NCSN objective: `venom.ncsn`
- Score SDE objectives and SDE definitions: `venom.score_sde`
- PFGM/PFGM++ objective: `venom.pfgm`
- flow matching, rectified flow, OT-CFM, stochastic interpolants: `venom.flow_matching`
- consistency, shortcut, MeanFlow, progressive distillation: `venom.one_step`
- fast samplers: `venom.samplers`
- MNIST command-line examples: `venom.train_mnist`, `venom.sample_mnist`

Images are normalized to `[-1, 1]` during training and converted back to `[0, 1]`
when saving grids.

## References

This project is inspired by the following papers. Venue labels use the archival
publication where available; recent preprints are marked as arXiv.

- **DDPM**: Ho, Jain, and Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
- **Improved DDPM**: Nichol and Dhariwal. *Improved Denoising Diffusion Probabilistic Models*. ICML 2021.
- **ADM / Guided Diffusion**: Dhariwal and Nichol. *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS 2021.
- **DiT**: Peebles and Xie. *Scalable Diffusion Models with Transformers*. ICCV 2023.
- **NCSN**: Song and Ermon. *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS 2019.
- **NCSNv2**: Song and Ermon. *Improved Techniques for Training Score-Based Generative Models*. NeurIPS 2020.
- **Score SDE**: Song et al. *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021.
- **EDM**: Karras et al. *Elucidating the Design Space of Diffusion-Based Generative Models*. NeurIPS 2022.
- **PFGM**: Xu, Liu, Tegmark, and Jaakkola. *Poisson Flow Generative Models*. NeurIPS 2022.
- **PFGM++**: Xu et al. *PFGM++: Unlocking the Potential of Physics-Inspired Generative Models*. ICML 2023.
- **Rectified Flow**: Liu, Gong, and Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR 2023.
- **Flow Matching**: Lipman et al. *Flow Matching for Generative Modeling*. ICLR 2023.
- **Conditional Flow Matching / OT-CFM**: Tong et al. *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*. TMLR 2024; also presented at ICML 2023 Frontiers4LCD Workshop.
- **Stochastic Interpolants**: Albergo, Boffi, and Vanden-Eijnden. *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*. JMLR 2025.
- **Progressive Distillation**: Salimans and Ho. *Progressive Distillation for Fast Sampling of Diffusion Models*. ICLR 2022.
- **Consistency Models**: Song, Dhariwal, Chen, and Sutskever. *Consistency Models*. ICML 2023.
- **Shortcut Models**: Frans, Hafner, Levine, and Abbeel. *One Step Diffusion via Shortcut Models*. ICLR 2025 Oral.
- **MeanFlow**: Geng et al. *Mean Flows for One-step Generative Modeling*. arXiv 2025.
