"""Teste de força bruta dos hiperparâmetros de um SVM para a base Digits."""

from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    dados = load_digits()
    X, y = dados.data, dados.target

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = make_pipeline(StandardScaler(), SVC())

    # Grade testada por força bruta: todas as combinações são avaliadas.
    grade = {
        "svc__kernel": ["linear", "rbf", "poly"],
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", 0.01, 0.001],
    }

    validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    busca = GridSearchCV(
        pipeline,
        param_grid=grade,
        scoring="accuracy",
        cv=validacao,
        n_jobs=-1,
        return_train_score=False,
    )

    inicio = perf_counter()
    busca.fit(X_treino, y_treino)
    tempo = perf_counter() - inicio

    melhor_modelo = busca.best_estimator_
    previsoes = melhor_modelo.predict(X_teste)
    acuracia_teste = accuracy_score(y_teste, previsoes)

    saida = Path(__file__).resolve().parent / "resultados_svm_bruteforce"
    saida.mkdir(exist_ok=True)

    quantidade_combinacoes = len(busca.cv_results_["params"])
    relatorio = (
        "SVM — teste de força bruta de hiperparâmetros\n"
        f"Base: {len(X)} imagens; {X.shape[1]} características por imagem\n"
        f"Treinamento: {len(X_treino)} | Teste: {len(X_teste)}\n"
        f"Combinações testadas: {quantidade_combinacoes}\n"
        f"Validação cruzada: {validacao.n_splits} folds\n"
        f"Tempo total: {tempo:.2f} segundos\n\n"
        f"Melhores parâmetros: {busca.best_params_}\n"
        f"Melhor acurácia na validação: {busca.best_score_:.2%}\n"
        f"Acurácia no conjunto de teste: {acuracia_teste:.2%}\n\n"
        + classification_report(y_teste, previsoes, digits=4, zero_division=0)
    )
    print(relatorio)
    (saida / "relatorio_bruteforce.txt").write_text(relatorio, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(y_teste, previsoes, ax=ax, cmap="Blues", colorbar=False)
    ax.set(title="SVM (melhor combinação): matriz de confusão",
           xlabel="Número previsto", ylabel="Número verdadeiro")
    fig.tight_layout()
    fig.savefig(saida / "matriz_confusao_bruteforce.png", dpi=180)
    plt.close(fig)

    ranking = sorted(
        zip(busca.cv_results_["mean_test_score"], busca.cv_results_["params"]),
        key=lambda item: item[0],
        reverse=True,
    )
    ranking_texto = "Ranking das combinações por acurácia média na validação cruzada\n"
    ranking_texto += "=" * 70 + "\n"
    for posicao, (score, parametros) in enumerate(ranking, start=1):
        ranking_texto += f"{posicao:02d}. {score:.4f} — {parametros}\n"
    (saida / "ranking_combinacoes.txt").write_text(ranking_texto, encoding="utf-8")

    print(f"Resultados salvos em: {saida}")


if __name__ == "__main__":
    main()
