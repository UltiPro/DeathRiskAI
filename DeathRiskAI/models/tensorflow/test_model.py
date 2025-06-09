import pandas as pd
from tensorflow_model import TensorflowModel
from sklearn.metrics import classification_report

if __name__ == "__main__":
    # Load test data
    X_test = pd.read_parquet("./../train_test_data/X_test.parquet")
    Y_test = pd.read_parquet("./../train_test_data/Y_test.parquet").squeeze()

    # Load final model
    model = TensorflowModel(model_name="final_model")
    model.load("models/final_model.keras")

    # Predict
    predictions = model.predict(X_test, threshold=0.3)
    proba = model.predict_proba(X_test)

    # Save predictions
    results = pd.DataFrame({
        "true": Y_test,
        "pred": predictions,
        "proba": proba
    })
    results.to_csv("results/test_predictions.csv", index=False)

    # Classification report
    report = classification_report(Y_test, predictions, output_dict=True)
    print("\nClassification Report on Test Data:")
    print(classification_report(Y_test, predictions))

    pd.DataFrame(report).transpose().to_csv("results/test_metrics.csv", index=True)

    print("\n✅ Test predictions and metrics saved to results/")
