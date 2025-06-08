#!/bin/bash
echo "=============================================="
echo "  Step 1: Tuning hiperparametrów (cross-val)  "
echo "=============================================="
#python3 tune_cv_model.py

echo "=============================================="
echo "  Step 2: Trenowanie finalnego modelu         "
echo "=============================================="
#python3 train_cv_model.py

echo "=============================================="
echo "  Step 3: Testowanie finalnego modelu         "
echo "=============================================="
python3 test_model.py

echo "=============================================="
echo "  Wszystko gotowe! Wyniki zapisane!           "
echo "=============================================="
