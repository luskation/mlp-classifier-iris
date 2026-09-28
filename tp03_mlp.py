# ==============================================================
# TP03 - MLPClassifier (Iris e Wine) + comparação com KNN (TP01)
# ==============================================================
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import (
    ConfusionMatrixDisplay,
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

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=base.target_names, cmap="Blues"
        )
        disp.ax_.set_title(f"Matriz de Confusão - {nome_modelo} - {nome_base}")
        plt.tight_layout()
        plt.savefig(f"figuras/cm_{nome_base.lower()}_{nome_modelo.lower()}.png",
                    dpi=150)
        plt.show()

# --------------------------------------------------------------
# 4. Tabela comparativa
# --------------------------------------------------------------
df = pd.DataFrame(resultados)
pd.set_option("display.float_format", "{:.4f}".format)
print("\n", df.to_string(index=False))
df.to_csv("resultados.csv", index=False)

# --------------------------------------------------------------
# 5. Gráfico comparativo das métricas
# --------------------------------------------------------------
metricas = ["Acurácia", "Precisão", "Revocação"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, nome_base in zip(axes, BASES):
    (df[df["Base"] == nome_base]
        .set_index("Modelo")[metricas].T
        .plot.bar(ax=ax, rot=0))
    ax.set_title(nome_base)
    ax.set_ylim(0.8, 1.01)
    ax.set_ylabel("Valor")
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("KNN (TP01) x MLPClassifier (TP03)")
plt.tight_layout()
plt.savefig("figuras/comparacao_metricas.png", dpi=150)
plt.show()
