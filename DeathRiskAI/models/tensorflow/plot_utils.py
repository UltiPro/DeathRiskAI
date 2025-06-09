import matplotlib.pyplot as plt
import os


def plot_training_history(history, name="model"):
    """
    Plots training and validation loss and saves to PNG and TXT.
    """
    os.makedirs("visualizations", exist_ok=True)

    # Plot loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Training and Validation Loss for {name}")
    plt.savefig(f"visualizations/{name}_loss.png")
    plt.close()

    # Save history as TXT
    with open(f"visualizations/{name}_history.txt", "w") as f:
        for key, values in history.history.items():
            f.write(f"{key}: {values}\n")


def plot_tuner_results(tuner, name="tuner"):
    """
    Plots the results summary of Keras Tuner search and saves to TXT.
    """
    pass
    # os.makedirs("results", exist_ok=True)
    # Save results summary to a text file
    # with open(f"results/{name}_summary.txt", "w") as f:
    # f.write(tuner.results_summary())
