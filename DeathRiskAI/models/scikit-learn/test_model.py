import os
import json
import pandas as pd
from utils import save_metrics_table, save_metrics_plot
from sklearn.metrics import classification_report
from sklearn.externals import joblib

# Załaduj dane testowe
print("🔄 Loading test data for model evaluation...")
X_test = pd.read_parquet("./trainval_test_data/X_test.parquet")
Y_test = pd.read_parquet("./trainval_test_data/Y_test.parquet").squeeze()

# Załaduj wytrenowany model
print("🔄 Loading the trained model from file...")
model = joblib.load("models/random_forest_model.pkl")

# Predykcje na zbiorze testowym
print("🔄 Making predictions on the test set...")
Y_pred = model.predict(X_test)

# Ocena modelu
print("🔄 Evaluating the model...")
report = classification_report(Y_test, Y_pred, output_dict=True)
print(report)

# Zapisz metryki
print("💾 Saving the classification report...")
save_metrics_table(report)

# Zapisz wykresy metryk
print("📈 Plotting and 💾 saving the metrics plot...")
save_metrics_plot(report)

print("✅ Model evaluation completed successfully!")
