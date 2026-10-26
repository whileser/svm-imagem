"""
Base pública: Digits (UCI), distribuída com o scikit-learn.
Fonte: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    dados = load_digits()
    X, y = dados.data, dados.target

  
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

   
    modelo = make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")
    )
    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, previsoes)
    relatorio = (
        "SVM — reconhecimento de números escritos à mão\n"
        f"Base: {len(X)} imagens; {X.shape[1]} características por imagem\n"
        f"Treinamento: {len(X_treino)} | Teste: {len(X_teste)}\n"
        f"Acurácia no teste: {acuracia:.2%}\n\n"
        "Precisão: proporção de previsões da classe que estavam corretas.\n"
        "Recall: proporção de exemplos da classe que foram identificados.\n"
        "F1: média harmônica entre precisão e recall.\n"
        "Support: quantidade de exemplos da classe no teste.\n\n"
        + classification_report(y_teste, previsoes, digits=4, zero_division=0)
        + f"\nTotal de vetores de suporte: {modelo[-1].n_support_.sum()}\n"
    )
    print(relatorio)

    saida = Path(__file__).resolve().parent / "resultados_svm"
    saida.mkdir(exist_ok=True)
    (saida / "relatorio.txt").write_text(relatorio, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_teste, previsoes, ax=ax, cmap="Blues", colorbar=False
    )
    ax.set(title="SVM: matriz de confusão", xlabel="Número previsto",
           ylabel="Número verdadeiro")
    fig.tight_layout()
    fig.savefig(saida / "matriz_confusao.png", dpi=180)
    plt.close(fig)

    fig, eixos = plt.subplots(3, 4, figsize=(9, 7))
    for i, ax in enumerate(eixos.flat):
        ax.imshow(X_teste[i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
        cor = "green" if previsoes[i] == y_teste[i] else "red"
        ax.set_title(f"Real: {y_teste[i]} | Previsto: {previsoes[i]}", color=cor)
        ax.axis("off")
    fig.suptitle("Reconhecimento de números escritos à mão com SVM")
    fig.tight_layout()
    fig.savefig(saida / "exemplos_previsoes.png", dpi=180)
    plt.close(fig)
    print(f"Relatório e gráficos salvos em: {saida}")


if __name__ == "__main__":
    main()
