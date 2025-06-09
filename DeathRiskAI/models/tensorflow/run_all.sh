#!/bin/bash

echo "=============================================="
echo "  Step 1: Tuning hyperparameters              "
echo "=============================================="
python3 tune_model.py

echo "=============================================="
echo "  Step 2: Training final model                "
echo "=============================================="
python3 train_model.py

echo "=============================================="
echo "  Step 3: Testing final model                 "
echo "=============================================="
python3 test_model.py

echo "=============================================="
echo "  All steps complete!                         "
echo "=============================================="
