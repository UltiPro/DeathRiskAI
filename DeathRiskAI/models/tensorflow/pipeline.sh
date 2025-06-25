#!/bin/bash

export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2

echo "=============================================="
echo "  Step 1: Tuning hyperparameters              "
echo "=============================================="
python tune_model.py

echo "=============================================="
echo "  Step 2: Training final model                "
echo "=============================================="
python train_model.py

echo "=============================================="
echo "  Step 3: Testing final model                 "
echo "=============================================="
python test_model.py

echo "=============================================="
echo "  All steps complete!                         "
echo "=============================================="
