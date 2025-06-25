import os
import io
import sys
import matplotlib.pyplot as plt


def plot_tuner_results(tuner, name="tuner"):
    """
    Saves the summary of Keras Tuner search to a text file.
    """
    os.makedirs("results", exist_ok=True)

    # Redirect stdout to capture the summary output
    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer

    # Generate the summary of the tuner results
    try:
        tuner.results_summary()
    except Exception as e:
        print(f"Error during tuner results summary: {e}")
        print("Please ensure that the Keras Tuner is properly configured and has completed the search.")
    finally:
        # Reset stdout to its original state
        sys.stdout = sys_stdout

    # Save the captured output to a text file
    with open(f"results/{name}_summary.txt", "w") as f:
        f.write(buffer.getvalue())


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

    # Save history to a text file
    with open(f"visualizations/{name}_history.txt", "w") as f:
        for key, values in history.history.items():
            f.write(f"{key}: {values}\n")
