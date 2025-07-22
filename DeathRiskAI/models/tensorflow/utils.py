import io
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_tuner import Tuner
from keras.utils import plot_model

from feature_transformation import RANDOM_SEED as RND_SEED

RANDOM_SEED = RND_SEED


def save_tuner_results(tuner: Tuner, name: str = "tuner") -> None:
    """
    Saves the summary of Keras Tuner search to a text file.
    """
    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)

    # Redirect stdout to capture the summary output
    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer

    try:
        # Generate the summary of the tuner results
        tuner.results_summary()
    except Exception as e:
        print(f"🛑 Error during tuner results summary: {e}")
        return
    finally:
        # Reset stdout to its original state
        sys.stdout = sys_stdout

    # Save the captured output to a text file
    with open(f"results/{name}_summary.txt", "w") as f:
        f.write(buffer.getvalue())


def save_model_visualization(model: tf.keras.Model, name: str = "model") -> None:
    """
    Visualizes the structure of the Keras model and saves it as an image.
    """
    # Ensure the visualizations directory exists
    os.makedirs("visualizations", exist_ok=True)

    # Plot the model structure and save it to a file
    plot_model(
        model,
        to_file=f"visualizations/{name}_structure.png",
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        expand_nested=True,
        rankdir="TB",
        dpi=300,
    )


def save_training_history(history: tf.keras.callbacks.History, name: str = "model") -> None:
    """
    Plots training and validation loss and saves to PNG and TXT.
    """
    # Ensure the visualizations directory exists
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
