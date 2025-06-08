import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_history(history, name="model"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"{name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title(f"{name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{name}_training_history.png")
    plt.show()


def plot_tuner_results(tuner):
    tuner.results_summary()

    trials = tuner.oracle.get_best_trials(num_trials=10)
    accuracies = [trial.score for trial in trials]
    trial_ids = [trial.trial_id for trial in trials]

    plt.figure(figsize=(8, 5))
    plt.bar(trial_ids, accuracies)
    plt.xlabel("Trial ID")
    plt.ylabel("Validation Accuracy")
    plt.title("Top KerasTuner Trials")
    plt.tight_layout()
    plt.savefig("tuner_results.png")
    plt.show()
