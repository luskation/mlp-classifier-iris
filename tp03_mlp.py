# ==============================================================
# TP03 - MLPClassifier (Iris e Wine) + comparação com KNN (TP01)
# ==============================================================
import os
import time

from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
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


# --------------------------------------------------------------
# 3. Treino, avaliação e matriz de confusão
# --------------------------------------------------------------
resultados = []

for nome_base, base in BASES.items():
    X, y = base.data, base.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\n{'=' * 60}\nBase: {nome_base}  "
          f"(treino={len(X_train)}, teste={len(X_test)})\n{'=' * 60}")

    for nome_modelo, modelo in criar_modelos().items():
        inicio = time.perf_counter()
        modelo.fit(X_train, y_train)
        tempo = time.perf_counter() - inicio

        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        cv = cross_val_score(criar_modelos()[nome_modelo], X, y, cv=5)

        print(f"\n--- {nome_modelo} ---")
        print(classification_report(y_test, y_pred,
                                    target_names=base.target_names, digits=4))

        resultados.append({
            "Base": nome_base,
            "Modelo": nome_modelo,
            "Acurácia": acc,
            "Precisão": prec,
            "Revocação": rec,
            "CV acc (média)": cv.mean(),
            "CV acc (desvio)": cv.std(),
            "Tempo treino (s)": tempo,
        })
