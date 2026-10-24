import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "../Assignment-3")
import reader

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def prepare_data():

    expression, metadata = reader.load_data("../SRP092402")

    groups = ["tb subjects", "healthy controls"]

    metadata = metadata[metadata["refinebio_disease"].isin(groups)].copy()

    metadata = metadata.set_index("refinebio_accession_code")

    sample_ids = [sample for sample in expression.columns if sample in metadata.index]

    expression = expression[sample_ids]
    metadata = metadata.loc[sample_ids]

    expression = np.log2(expression + 1)

    labels = (metadata["refinebio_disease"] == "tb subjects").astype(int)

    print("Expression matrix:", expression.shape)
    print("Samples:", len(labels))

    print("\nDisease groups:")
    print(metadata["refinebio_disease"].value_counts())

    return expression, metadata, labels


def create_clusters(expression):

    # Reproduce Assignment 3 K-means with 1000 genes
    gene_variance = expression.var(axis=1)
    top_genes = gene_variance.nlargest(1000).index

    X = expression.loc[top_genes].T.values

    model = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=20,
    )

    clusters = model.fit_predict(X)

    print("\nK-means cluster sizes:")
    print(pd.Series(clusters).value_counts().sort_index())

    return pd.Series(
        clusters,
        index=expression.columns,
        name="cluster",
    )


def train_model(expression, labels, n_genes, name, train_ids, test_ids):

    X = expression.T
    y = pd.Series(labels, index=X.index)

    X_train = X.loc[train_ids]
    X_test = X.loc[test_ids]
    y_train = y.loc[train_ids]
    y_test = y.loc[test_ids]

    # genes used for training the model
    gene_variance = X_train.var(axis=0)
    top_genes = gene_variance.nlargest(n_genes).index

    X_train = X_train[top_genes]
    X_test = X_test[top_genes]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=5000,
                    solver="liblinear",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print(f"\n{name} - {n_genes} genes")
    print("Accuracy:", accuracy)
    print("AUC:", auc)

    prediction_results = pd.DataFrame(
        {
            "sample": X_test.index,
            "actual": y_test.values,
            "predicted": predictions,
            "probability": probabilities,
        }
    )

    prediction_results.to_csv(
        f"out/{name}_{n_genes}_predictions.csv",
        index=False,
    )

    coefficients = pd.DataFrame(
        {
            "gene": top_genes,
            "coefficient": model.named_steps["logistic"].coef_[0],
        }
    )

    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()

    coefficients = coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    coefficients.to_csv(
        f"out/{name}_{n_genes}_genes.csv",
        index=False,
    )

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
    )

    plt.title(f"{name} Logistic Regression ({n_genes} genes)")
    plt.tight_layout()
    plt.savefig(
        f"out/{name}_{n_genes}_roc.png",
        dpi=300,
    )
    plt.close()

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["0", "1"],
    )

    display.plot(cmap="Blues")

    plt.title(f"{name} ({n_genes} genes)")
    plt.tight_layout()
    plt.savefig(
        f"out/{name}_{n_genes}_confusion.png",
        dpi=300,
    )
    plt.close()

    return {
        "Target": name,
        "Genes": n_genes,
        "Accuracy": accuracy,
        "AUC": auc,
    }


def main():

    os.makedirs("out", exist_ok=True)

    expression, metadata, disease_labels = prepare_data()

    clusters = create_clusters(expression)

    train_ids, test_ids = train_test_split(
        expression.columns,
        test_size=0.2,
        random_state=42,
        stratify=disease_labels,
    )
    pd.DataFrame({"sample": test_ids}).to_csv(
        "out/test_sample_ids.csv",
        index=False,
    )

    sample_labels = pd.DataFrame(
        {
            "sample": expression.columns,
            "disease": metadata["refinebio_disease"].values,
            "disease_label": disease_labels.values,
            "cluster": clusters.values,
        }
    )

    sample_labels.to_csv(
        "out/sample_labels.csv",
        index=False,
    )

    results = []

    results.append(
        train_model(expression, disease_labels, 5000, "disease", train_ids, test_ids)
    )

    results.append(
        train_model(expression, clusters, 5000, "cluster", train_ids, test_ids)
    )

    for n_genes in [10, 100, 1000, 10000]:
        results.append(
            train_model(
                expression,
                disease_labels,
                n_genes,
                "disease",
                train_ids,
                test_ids,
            )
        )

        results.append(
            train_model(
                expression,
                clusters,
                n_genes,
                "cluster",
                train_ids,
                test_ids,
            )
        )
    results = pd.DataFrame(results)

    disease = pd.read_csv("out/disease_5000_predictions.csv")
    cluster = pd.read_csv("out/cluster_5000_predictions.csv")

    assert set(disease["sample"]) == set(cluster["sample"])

    print("Disease and cluster test samples match!")
    results.to_csv(
        "out/model_summary.csv",
        index=False,
    )

    print("\nLOGISTIC REGRESSION RESULTS")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
