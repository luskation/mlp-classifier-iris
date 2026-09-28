# ==============================================================
# TP03 - MLPClassifier (Iris e Wine) + comparação com KNN (TP01)
# ==============================================================
import os

from sklearn.datasets import load_iris, load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.3
K_KNN = 5  # usar o mesmo k do TP01
os.makedirs("figuras", exist_ok=True)

# --------------------------------------------------------------
# 1. Bases
# --------------------------------------------------------------
BASES = {"Iris": load_iris(), "Wine": load_wine()}


# --------------------------------------------------------------
# 2. Modelos (scaler dentro do pipeline -> sem data leakage)
# --------------------------------------------------------------
def criar_modelos():
    return {
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=K_KNN),
        ),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(10, 10),
                activation="relu",
                solver="adam",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    }
