import pandas as pd
import matplotlib.pyplot as plt


def save_metrics_table(report: dict) -> None:
    """
    Saves a pretty table of classification metrics to a text file.
    """
    with open("results/metrics_table.txt", "w", encoding="utf-8") as f:
        f.write("📊 Classification Report\n\n")
        f.write(
            "{:<15} {:<10} {:<10} {:<10} {:<10}\n".format(
                "Class", "Precision", "Recall", "F1-score", "Support"
            )
        )
        f.write("-" * 60 + "\n")
        for label in ["0", "1", "accuracy", "macro avg", "weighted avg"]:
            if label == "accuracy":
                support_sum = int(report["0"]["support"]) + int(report["1"]["support"])
                f.write(
                    "{:<15} {:<10} {:<10} {:<10.4f} {:<10}\n".format(
                        label, "", "", report["accuracy"], support_sum
                    )
                )
            else:
                row = report[label]
                f.write(
                    "{:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10}\n".format(
                        label,
                        row.get("precision", 0.0),
                        row.get("recall", 0.0),
                        row.get("f1-score", 0.0),
                        int(row.get("support", 0)),
                    )
                )


def save_metrics_plot(report: dict) -> None:
    """
    Saves a bar plot of precision, recall, and F1-score for each class.
    """
    pd.DataFrame(report).transpose().loc[["0", "1"], ["precision", "recall", "f1-score"]].plot(
        kind="bar", figsize=(8, 6)
    )
    plt.title("Precision, Recall, F1-score per Class")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results/metrics.png")
    plt.close()
