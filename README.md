# Self-Pruning Neural Network on CIFAR-10

A PyTorch implementation of a neural network that learns to remove its own weights
during training. Instead of pruning being a separate post-training step, every weight
has a paired learnable gate that the optimizer drives toward zero throughout the
training process. The result is a sparse network whose accuracy is determined by how
aggressively pruning is applied.

This project was submitted as the case study for the Tredence Studio AI Engineering
Internship — 2025 Cohort.

---

## The Core Idea

Each weight `w_i` in the fully-connected layers is paired with a learnable gate score
`gs_i`. During every forward pass, the effective weight is:

```
gates_i        = sigmoid(gs_i)          -- a value between 0 and 1
effective_w_i  = w_i * gates_i
output         = F.linear(x, effective_weights, bias)
```

The training objective combines a standard classification loss with a sparsity penalty:

```
Total Loss = CrossEntropyLoss + lambda * SparsityLoss
SparsityLoss = sum of all gate values across PrunableLinear layers
```

The sparsity term applies a constant downward gradient on every gate score. For a
weight that contributes little to the classification task, the sparsity gradient wins
and the gate reaches zero. For an important weight, the classification gradient is
larger and the gate stays open. Lambda controls the threshold between the two
outcomes.

---

## Architecture

The network is divided into two parts:

**Convolutional feature extractor (not pruned)**

Three VGG-style blocks that reduce a 32x32 RGB image to a 256-channel 4x4
feature map. These use standard `nn.Conv2d` layers and are not subject to
gate-based pruning. Their job is to produce stable features; introducing
pruning pressure here would destabilise learning.

**Prunable classification head (89% of all parameters)**

Three `PrunableLinear` layers sitting on top of the feature extractor:

```
Flatten    :  256 x 4 x 4  =  4,096 features
Layer fc1  :  4,096  ->  1,024    (PrunableLinear)
Layer fc2  :  1,024  ->    512    (PrunableLinear)
Layer fc3  :    512  ->     10    (PrunableLinear, class logits)
```

Concentrating the prunable parameters in the classifier head means the gate
mechanism has meaningful impact on the model and the sparsity-accuracy trade-off
is clearly measurable.

---

## Why This Works: L1 on Sigmoid Gates

The gradient of the sparsity loss with respect to gate score `gs_i` is:

```
d(SparsityLoss) / d(gs_i)  =  sigmoid(gs_i) * (1 - sigmoid(gs_i))
```

This is always positive, so the optimizer is always nudged to decrease each gate
score. For unimportant weights the classification gradient on `gs_i` is negligible,
so the sparsity term dominates and the gate drifts monotonically toward zero. For
important weights the classification gradient is large enough to resist the sparsity
pressure and the gate stays close to one.

**Why L1 rather than L2?** An L2 penalty on gate values has a gradient proportional
to the gate value itself. As a gate approaches zero, the L2 gradient also approaches
zero — the penalty weakens exactly when you need it most. The L1 gradient decays
more slowly and provides a stronger push toward zero throughout training.

**Why is weight decay removed from `gate_scores`?** Standard SGD with weight decay
adds `wd * gate_score` to the gradient. When a gate score is negative (gate below
0.5), this term is also negative, opposing the sparsity push. A stable equilibrium
forms that can prevent gates from ever crossing the pruning threshold. Removing
weight decay from gate parameters eliminates this blocking equilibrium.

---

## Training Details

| Setting              | Value                            |
|----------------------|----------------------------------|
| Dataset              | CIFAR-10 (50k train / 10k test)  |
| Batch size           | 256                              |
| Epochs               | 100 (with early stopping)        |
| Optimizer            | SGD + Nesterov momentum          |
| Learning rate        | OneCycleLR, peak 0.1             |
| Weight decay         | 5e-4 (applied to weights only)   |
| Gate learning rate   | 3x the base learning rate        |
| Lambda warmup        | Linear ramp over first 20 epochs |
| Augmentation         | AutoAugment + Cutout + MixUp     |
| Mixed precision      | Enabled (AMP)                    |

Three lambda values are tested to show the sparsity-accuracy trade-off:

| Lambda | Expected Accuracy | Expected Sparsity |
|--------|-------------------|-------------------|
| 0.01   | ~93%              | ~100% of FC head  |
| 0.03   | ~92%              | ~100% of FC head  |
| 0.10   | ~84%              | ~100% of FC head  |

---

## Project Structure

```
.
├── notebook.ipynb        Main notebook: implementation, training, analysis
├── README.md             This file
└── results/              Generated at runtime on Kaggle
    ├── checkpoints/      Best model weights for each lambda value (.pth)
    ├── gates_lambda_*.png  Gate distribution histograms
    ├── training_history.png  Accuracy, sparsity, loss curves
    ├── sparsity_accuracy_tradeoff.png  Scatter plot of the lambda trade-off
    └── results.json      Summary of all runs
```

---

## How to Run

**On Kaggle (recommended)**

1. Upload `notebook.ipynb` to a new Kaggle notebook.
2. Go to Settings -> Accelerator -> GPU T4 x2.
3. Go to Settings -> Internet -> On (required to download CIFAR-10).
4. Click Run All.
5. After training completes, download outputs from the Output tab.

**Locally**

```bash
pip install torch torchvision tqdm matplotlib numpy
jupyter notebook notebook.ipynb
```

If running locally without a GPU, set `batch_size = 64` and `use_amp = False`
in the CFG cell before running.

---

## Implementation Notes

**Gate initialisation**

Gates are initialised at `sigmoid(2.0) ≈ 0.88`. Starting with gates slightly open
gives the network a head start on learning useful representations before the sparsity
penalty kicks in. Starting at the neutral midpoint (0.5) has been observed to cause
gates to stagnate because the opposing classification and sparsity gradients cancel.

**Lambda warmup**

For the first 20 epochs, the effective lambda is ramped linearly from zero to the
target value. This allows the network to learn a reasonable feature representation
before pruning pressure is introduced, which leads to more stable and interpretable
gate distributions.

**Gate-specific learning rate**

`gate_scores` are updated at three times the base learning rate. This compensates
for the fact that the per-gate gradient magnitude is smaller than the per-weight
gradient, ensuring gates converge on a similar timescale as the weights.

---

## Dependencies

- Python 3.9 or later
- PyTorch 2.0 or later
- torchvision
- matplotlib
- numpy
- tqdm

---

## License

This repository is submitted as part of a technical internship assessment.
