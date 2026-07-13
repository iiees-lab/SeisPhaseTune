import pandas as pd
import matplotlib.pyplot as plt
##########################################################################

filepath = r"D:/Models/PhaseNet_2026-07-13T15-53-06/loss_2026-07-13T15-53-06.csv"
df = pd.read_csv(filepath)


# Global step
df["global_step"] = range(1, len(df) + 1)



fig, ax = plt.subplots(figsize=(10, 5))

# -------------------------
# Loss axis (left y-axis)
# -------------------------
ax.plot(
    df["global_step"],
    df["train_loss"],
    label="Train Loss"
)

test_df = df[df["test_loss"].notna()]

ax.scatter(
    test_df["global_step"],
    test_df["test_loss"],
    label="Test Loss",
    zorder=3,
    color='r',
)

ax.set_xlabel("Global Step")
ax.set_ylabel("Loss")
ax.grid(True, alpha=0.3)


# -------------------------
# Learning rate axis (right y-axis)
# -------------------------
ax_lr = ax.twinx()

ax_lr.step(
    df["global_step"],
    df["lr"],
    where="post",
    alpha=0.35,
    linewidth=2,
    label="Learning Rate",
    color='g'
)

ax_lr.set_yscale("log")
ax_lr.set_ylabel("Learning Rate")


# -------------------------
# Epoch labels on top axis
# -------------------------
ax_epoch = ax.twiny()
ax_epoch.set_xlim(ax.get_xlim())

epoch_start = (
    df.groupby(["stage", "epoch"])
      .first()["global_step"]
)

ax_epoch.set_xticks(epoch_start.values)
ax_epoch.set_xticklabels(
    [f"S{s}-E{e}" for s, e in epoch_start.index],
    rotation=45,
    ha="left"
)

ax_epoch.set_xlabel("Epoch")


# -------------------------
# Combine legends
# -------------------------
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_lr.get_legend_handles_labels()

ax.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right"
)


ax.set_yscale('log')
plt.tight_layout()
plt.show()
