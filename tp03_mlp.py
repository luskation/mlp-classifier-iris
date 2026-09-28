# ==============================================================
# TP03 - MLPClassifier (Iris e Wine) + comparação com KNN (TP01)
# ==============================================================
import os

from sklearn.datasets import load_iris, load_wine

RANDOM_STATE = 42
TEST_SIZE = 0.3
K_KNN = 5  # usar o mesmo k do TP01
os.makedirs("figuras", exist_ok=True)

# --------------------------------------------------------------
# 1. Bases
# --------------------------------------------------------------
BASES = {"Iris": load_iris(), "Wine": load_wine()}
