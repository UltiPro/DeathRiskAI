import os
import json
import pandas as pd
from sklearn.model_selection import GridSearchCV
from utils import save_grid_search_results
from scikit_model import SklearnModel
from sklearn.metrics import classification_report

# Upewnij się, że foldery do zapisów istnieją
os.makedirs("results", exist_ok=True)

# Załaduj dane
print("🔄 Loading trainval data for hyperparameter tuning...")
X_trainval = pd.read_parquet("./trainval_test_data/X_trainval.parquet")
Y_trainval = pd.read_parquet("./trainval_test_data/Y_trainval.parquet").squeeze()

# Określenie zakresu hiperparametrów do przeszukiwania
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

# Tworzymy obiekt modelu SklearnModel
model = SklearnModel(model_name="random_forest_model", random_seed=42)

# Tworzymy model RandomForestClassifier, który będzie używał SklearnModel
rf_model = model.build(config={}, input_dim=X_trainval.shape[1])

# Używamy GridSearchCV do strojenia hiperparametrów
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring="accuracy")

# Przeprowadzamy GridSearchCV
print("🔄 Starting hyperparameter tuning with GridSearchCV...")
grid_search.fit(X_trainval, Y_trainval)

# Zapisz wyniki grid search
print("💾 Saving grid search results...")
save_grid_search_results(grid_search)

# Zapisz najlepsze hiperparametry
best_params = grid_search.best_params_
print(f"✅ Best hyperparameters: {best_params}")
with open("results/best_hp.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("✅ Hyperparameter tuning completed successfully!")
