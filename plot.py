import os
import matplotlib.pyplot as plt

# =========================
# Data
# =========================
MAX_ROUND = 10
rounds = list(range(1, MAX_ROUND + 1))

results = {
    "G+/D+": {
        "Macro-F1": [0.6337, 0.7754, 0.7693, 0.7257, 0.7754, 0.7572, 0.7923, 0.8075, 0.7948, 0.8119],
        "ROC-AUC":  [0.8377, 0.8765, 0.8729, 0.8338, 0.8654, 0.9129, 0.8806, 0.8919, 0.8965, 0.9035],
        "Accuracy": [0.6592, 0.7823, 0.7761, 0.7264, 0.7761, 0.7674, 0.7960, 0.8085, 0.7985, 0.8134],
    },
    "G-/D+": {
        "Macro-F1": [0.7029, 0.7418, 0.7577, 0.7512, 0.7722, 0.8059, 0.7843, 0.7180, 0.8077, 0.8015],
        "ROC-AUC":  [0.7184, 0.8452, 0.8453, 0.8200, 0.8577, 0.8970, 0.8482, 0.8530, 0.8871, 0.8867],
        "Accuracy": [0.7226, 0.7488, 0.7587, 0.7525, 0.7774, 0.8060, 0.7861, 0.7239, 0.8085, 0.8022],
    },
    "G+/D-": {
        "Macro-F1": [0.3774, 0.7015, 0.6962, 0.3848, 0.3979, 0.7199, 0.7023, 0.6732, 0.6954, 0.7275],
        "ROC-AUC":  [0.7731, 0.7796, 0.7809, 0.7087, 0.7157, 0.7980, 0.7805, 0.7837, 0.7667, 0.7946],
        "Accuracy": [0.5187, 0.7139, 0.6965, 0.5187, 0.5261, 0.7201, 0.7027, 0.6928, 0.6965, 0.7289],
    },
    "G-/D-": {
        "Macro-F1": [0.6741, 0.5576, 0.5486, 0.6424, 0.5539, 0.7078, 0.6678, 0.6873, 0.6678, 0.6756],
        "ROC-AUC":  [0.7454, 0.7722, 0.7673, 0.7104, 0.7574, 0.7754, 0.7633, 0.7618, 0.7813, 0.7825],
        "Accuracy": [0.6766, 0.6007, 0.5933, 0.6443, 0.5970, 0.7114, 0.6704, 0.6978, 0.6716, 0.6803],
    },
}

metrics = ["Macro-F1", "ROC-AUC", "Accuracy"]

# =========================
# Output directory
# =========================
out_dir = "figs/rag_eval"
os.makedirs(out_dir, exist_ok=True)

# =========================
# Plot & save
# =========================
for metric in metrics:
    plt.figure(figsize=(7, 4))

    for setting, vals in results.items():
        plt.plot(rounds, vals[metric][:MAX_ROUND], label=setting)

    plt.xlabel("Round")
    plt.ylabel(metric)
    plt.title(f"Discriminator Evaluation with RAG - {metric}")

    # ✅ x 軸：每一 round 都顯示
    plt.xticks(rounds)
    plt.grid(axis="y")
    # ✅ legend 移到圖外（右上）
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0
    )

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # 留空間給 legend

    save_path = os.path.join(out_dir, f"{metric.lower().replace('-', '_')}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")