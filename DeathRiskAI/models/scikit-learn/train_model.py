import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import save_model, save_feature_importance, save_training_metrics
from scikit_model import SklearnModel
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# Upewnij się, że foldery do zapisów istnieją
os.makedirs("models", exist_ok=True)

# Załaduj dane do treningu
print("🔄 Loading trainval data for training the model...")
X_trainval = pd.read_parquet("./trainval_test_data/X_trainval.parquet")
Y_trainval = pd.read_parquet("./trainval_test_data/Y_trainval.parquet").squeeze()

# Załaduj najlepsze hiperparametry
print("🔄 Loading the best hyperparameters from results/best_hp.json...")
with open("results/best_hp.json", "r") as f:
    best_hp = json.load(f)

# Podziel dane na treningowe i walidacyjne
print("🔄 Splitting trainval data into training and validation sets...")
X_train, X_val, Y_train, Y_val = train_test_split(
    X_trainval, Y_trainval, test_size=0.1, stratify=Y_trainval, random_state=42
)

# Zastosowanie SMOTE do oversamplingu klasy mniejszościowej
print("🔄 Applying SMOTE oversampling to the training set...")
smote = SMOTE(random_state=42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)

# Budowanie modelu SklearnModel z najlepszymi hiperparametrami
print("🔄 Building the model with the best hyperparameters...")
model = SklearnModel(model_name="random_forest_model", random_seed=42)
rf_model = model.build(config=best_hp, input_dim=X_train_smote.shape[1])

# Trening modelu
print("🔄 Training the final model...")
rf_model.fit(X_train_smote, Y_train_smote)

# Ewaluacja modelu na zbiorze walidacyjnym
print("🔄 Evaluating the model on the validation set...")
Y_val_pred = rf_model.predict(X_val)
print(classification_report(Y_val, Y_val_pred))

# Wizualizacja i zapis ważności cech
print("📈 Visualizing and 💾 saving the feature importance...")
save_feature_importance(rf_model, X_train.columns, name="final_model")

# Zapisz model
print("💾 Saving the final model...")
save_model(rf_model, "models/random_forest_model.pkl")

# Zapisz metryki treningowe
print("💾 Saving the training metrics...")
training_metrics = classification_report(Y_val, Y_val_pred, output_dict=True)
save_training_metrics(pd.DataFrame(training_metrics).transpose(), name="final_model")

print("✅ Model training completed successfully!")
