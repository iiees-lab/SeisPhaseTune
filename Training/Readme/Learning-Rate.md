# Training Stage Strategy: Continuous vs. Staged Training

By controlling the number of entries in the `train_stages` list in `Parameters-cfg.yml`, the training loop in `02_01-Training.py` naturally supports two different optimization behaviors — **Continuous Training** and **Staged Training** — with zero code changes required to switch between them.

Each entry in `train_stages` is a fully self-contained block: it defines its own `optimizer`, `lr_scheduler`, and `early_stopping`, each given as a `cls` (a fully-qualified class path) plus that class's constructor arguments. `SeisRoutine.config.ObjectFactory.create` instantiates these dynamically at the start of every stage — which is also the mechanism behind the reset behavior described below.

---

## 🛠️ Strategy 1: Continuous Training (Momentum Retention)

### Concept

In standard deep learning pipelines, it's often desirable to initialize the optimizer exactly once and let it maintain its internal history (first and second moments of the gradients, for Adam) across the entire training duration, for a smooth trajectory through the loss landscape.

### Configuration Behavior

When `train_stages` contains a **single entry**, the outer stage loop executes exactly once. The optimizer, scheduler, and early-stopping object are created once and persist through all of that stage's epochs. Learning-rate adjustment is left entirely to whatever `lr_scheduler` you configure (typically `ReduceLROnPlateau`, driven by validation loss).

### YAML Configuration Example

```yaml
train_stages:
  - name: stage1
    optimizer:
        cls: torch.optim.Adam
        lr: 1.0e-3
    epochs: 100
    lr_scheduler:
        cls: torch.optim.lr_scheduler.ReduceLROnPlateau
        mode: min
        factor: 0.2
        patience: 5
        threshold: 1.0e-4
        threshold_mode: rel
        cooldown: 0
        min_lr: 0
        eps: 1.0e-08
    early_stopping:
        cls: SeisRoutine.seisbench.training.EarlyStopping
        patience: 10
        min_improvement_percent: 10
```

### Use Case

- Recommended for production training, final model fine-tuning, and long-running schedules where steady convergence is preferred.
- Effective when computational resources are limited, since it avoids destabilizing gradient tracking.

---

## ⚡ Strategy 2: Staged Training (Forced Optimizer Reset)

### Concept

When a model gets trapped in a sub-optimal local minimum or a flat plateau, lowering the learning rate within the same optimizer sometimes isn't enough to break free, because the accumulated momentum history keeps dragging the weights in the old direction.

### Configuration Behavior

When `train_stages` contains **multiple entries**, moving to the next entry forces a **hard reset**: a brand-new optimizer, scheduler, and early-stopping object are instantiated for that stage, wiping all previous momentum, variance tracking, and patience counters. This isn't special-cased logic — it's a direct consequence of `ObjectFactory.create` being called fresh at the top of every stage in the loop.

Because each stage carries its own `cls`, you're not limited to changing the learning rate between stages — the optimizer type, the scheduler type, and the early-stopping rule can all change too.

### YAML Configuration Example

```yaml
train_stages:
  - name: stage0_explore
    optimizer:
        cls: torch.optim.Adam
        lr: 1.0e-2
    epochs: 10
    lr_scheduler:
        cls: torch.optim.lr_scheduler.ReduceLROnPlateau
        mode: min
        factor: 0.2
        patience: 3
        threshold: 1.0e-4
        threshold_mode: rel
        cooldown: 0
        min_lr: 0
        eps: 1.0e-08
    early_stopping:
        cls: SeisRoutine.seisbench.training.EarlyStopping
        patience: 5
        min_improvement_percent: 10

  - name: stage1_stabilize
    optimizer:
        cls: torch.optim.Adam
        lr: 1.0e-3
    epochs: 20
    lr_scheduler:
        cls: torch.optim.lr_scheduler.ReduceLROnPlateau
        mode: min
        factor: 0.2
        patience: 3
        threshold: 1.0e-4
        threshold_mode: rel
        cooldown: 0
        min_lr: 0
        eps: 1.0e-08
    early_stopping:
        cls: SeisRoutine.seisbench.training.EarlyStopping
        patience: 10
        min_improvement_percent: 10

  - name: stage2_finetune
    optimizer:
        cls: torch.optim.SGD
        lr: 1.0e-4
        momentum: 0.9
    epochs: 20
    lr_scheduler:
        cls: torch.optim.lr_scheduler.CosineAnnealingLR
        T_max: 20
        eta_min: 1.0e-6
    early_stopping:
        cls: SeisRoutine.seisbench.training.EarlyStopping
        patience: 10
        min_improvement_percent: 5
```

Stage 2 above swaps to `SGD` with momentum and `CosineAnnealingLR` — a fully different optimization recipe, not just a different number.

### Use Case

- Useful when experimenting with highly non-convex loss landscapes or noisy signal datasets (such as seismic waveforms).
- Functions like a manual **Warm Restart**, pushing the model to explore new regions of parameter space.
- Lets you pair a coarse, high-LR optimizer for early exploration with a fine-grained, low-LR optimizer for late-stage convergence, in one config file.

---

## Behavior Notes

- **Checkpointing is global, not per-stage.** `best_val_loss` is initialized once, before the stage loop, so `best.ckpt` tracks the best validation loss across the *entire* run. If a new stage starts at a higher loss than the previous stage ended at (e.g. because of a higher restart LR), it won't overwrite `best.ckpt` until it beats the prior best.
- **Early stopping resets per stage.** Since a new `EarlyStopping` instance is created for each stage, a stage that was close to triggering early stopping does not carry that state into the next stage — every stage gets a full patience budget.
- **The epoch that triggers early stopping is not logged.** The `break` on early stopping happens before the per-batch logging block, so that epoch's losses won't appear in `loss_{model.version_str}.csv` or the loss plot — only completed epochs are recorded.

## Summary of Benefits

1. **Zero Code Changes:** Shift between a staged scheduling experiment and a long, continuous training run purely by adding or removing entries in `train_stages`.
2. **Experiment Logging:** The training loop captures `stage`, `epoch`, and `batch` indices automatically, writing batch-by-batch loss logs (`loss_{model.version_str}.csv`) regardless of which strategy is active.
3. **Resource Friendly:** Quick adjustments and dry-runs (e.g., small epoch counts across multiple stages, as in the current `Parameters-cfg.yml`) let you verify execution stability before committing to a full-scale run.