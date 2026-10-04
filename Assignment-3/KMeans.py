import reader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def main():

    expression, metadata = reader.load_data()

    groups = ["tb subjects", "healthy controls"]
    metadata = metadata[metadata["refinebio_disease"].isin(groups)].copy()

    metadata = metadata.set_index("refinebio_accession_code")

    sample_ids = [sample for sample in expression.columns if sample in metadata.index]

    expression = expression[sample_ids]
    metadata = metadata.loc[sample_ids]

    expression = np.log2(expression + 1)

    gene_variance = expression.var(axis=1)
    top_genes = gene_variance.nlargest(5000).index
    expression = expression.loc[top_genes]

    X = expression.T.values

    print("K-means input shape:", X.shape)

    Path("figs").mkdir(exist_ok=True)

    # ONLY for visualization.
    pca = PCA(n_components=2)
    X_plot = pca.fit_transform(X)

    for k in [2, 3, 4, 5]:

        model = KMeans(n_clusters=k, random_state=42, n_init=20)

        clusters = model.fit_predict(X)

        print(f"\nk = {k}")
        print("Cluster counts:")

        unique, counts = np.unique(clusters, return_counts=True)

        for cluster, count in zip(unique, counts):
            print(f"Cluster {cluster}: {count} samples")

        plt.figure(figsize=(8, 6))

        for cluster in range(k):

            mask = clusters == cluster

            plt.scatter(
                X_plot[mask, 0],
                X_plot[mask, 1],
                label=f"Cluster {cluster}",
                alpha=0.7,
                s=25,
            )

        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")

        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")

        plt.title(f"K-means Clusters Using 5,000 Most Variable Genes (k={k})")

        plt.legend()

        plt.tight_layout()
        plt.savefig(f"figs/kmeans_k{k}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
