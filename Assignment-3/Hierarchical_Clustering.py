# Following https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering-with-scikit-learn/

import reader
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import itertools

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency

from sklearn.decomposition import PCA

gene_count = 5000


def chi_square_clustering_comparison(labels1, labels2):
    # Compare two clustering results using chi-square test of independence.
    contingency_table = pd.crosstab(labels1, labels2)
    chi2_stat, p_value, dof, _ = chi2_contingency(contingency_table)
    return chi2_stat, p_value, dof


def run_gene_count_sensitivity(expression, metadata):
    # Run clustering with different gene counts and perform chi-square tests.
    gene_counts = [10, 100, 1000, 10000]
    n_clusters = 2

    clustering_results = {}

    print("Gene Count Sensitivity Analysis")

    for n_genes in gene_counts:
        gene_variance = expression.var(axis=1)
        top_genes = gene_variance.nlargest(n_genes).index
        expr_subset = expression.loc[top_genes]

        X = expr_subset.T.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        hc_model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = hc_model.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, labels)

        clustering_results[n_genes] = {"labels": labels, "silhouette": sil_score}

        print(f"\n{n_genes} genes - Silhouette Score: {sil_score:.4f}")
        print(f"Cluster distribution: {pd.Series(labels).value_counts().to_dict()}")

    # Perform chi-square tests on all pairs
    chi_square_results = []
    gene_counts_sorted = sorted(gene_counts)

    print("Chi-Square Test Results (k=2 clusters)")

    for genes1, genes2 in itertools.combinations(gene_counts_sorted, 2):
        labels1 = clustering_results[genes1]["labels"]
        labels2 = clustering_results[genes2]["labels"]

        chi2_stat, p_value, dof = chi_square_clustering_comparison(labels1, labels2)

        chi_square_results.append(
            {
                "Method 1": f"{genes1} genes",
                "Method 2": f"{genes2} genes",
                "Chi-square Statistic": round(chi2_stat, 4),
                "P-value": round(p_value, 6),
                "Degrees of Freedom": dof,
            }
        )

    results_df = pd.DataFrame(chi_square_results)
    print("\n" + results_df.to_string(index=False))

    return results_df, clustering_results


def main():
    expression, metadata = reader.load_data()

    groups = ["tb subjects", "healthy controls"]
    metadata = metadata[metadata["refinebio_disease"].isin(groups)].copy()

    print(
        "Count TB Subjects: ",
        metadata[metadata["refinebio_disease"] == "tb subjects"].shape[0],
    )

    print(
        "Count Healthy Subjects: ",
        metadata[metadata["refinebio_disease"] == "healthy controls"].shape[0],
    )

    metadata = metadata.set_index("refinebio_accession_code")

    sample_ids = [sample for sample in expression.columns if sample in metadata.index]

    expression = expression[sample_ids]
    metadata = metadata.loc[sample_ids]

    expression = np.log2(expression + 1)

    # Run gene count sensitivity analysis
    chi_square_df, _ = run_gene_count_sensitivity(expression, metadata)

    gene_variance = expression.var(axis=1)
    top_genes = gene_variance.nlargest(gene_count).index
    expression = expression.loc[top_genes]

    X = expression.T.values

    print("\n\nK-means input shape:", X.shape)

    Path("figs").mkdir(exist_ok=True)

    # Normalizing the input features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Plotting the dendrogram of the linked data.
    # I used this to arrive at a reasonable n_clusters estimate for the clustering model later.
    linked = linkage(X_scaled, method="ward")

    plt.figure(figsize=(12, 6))
    dendrogram(linked, truncate_mode="lastp", p=30)

    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Cluster Size")
    plt.ylabel("Distance")
    plt.savefig(
        "figs/hierarchical_clustering_dendrogram.png",
        dpi=300,
    )

    # Visualization
    pca = PCA(n_components=2)
    X_plot = pca.fit_transform(X)

    for clusters_estimate in range(2, 8):
        hc_model = AgglomerativeClustering(n_clusters=clusters_estimate, linkage="ward")

        # Fits the data to the clustering model with our estimated n_clusters value.
        cluster_labels = hc_model.fit_predict(X_scaled)

        # Provide a quick view of the distribution.
        print(pd.Series(cluster_labels).value_counts())

        # Evaluating cluster quality.
        sil_score = silhouette_score(X_scaled, cluster_labels)
        print(f"{clusters_estimate} Clusters Score: {round(sil_score, 2)}")

        plt.figure(figsize=(8, 6))

        for cluster in range(clusters_estimate):

            mask = cluster_labels == cluster

            plt.scatter(
                X_plot[mask, 0],
                X_plot[mask, 1],
                label=f"Cluster {cluster}",
                alpha=0.7,
                s=25,
            )

        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")

        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")

        plt.title(
            f"Hierarchical Clusters Using {gene_count} Most Variable Genes (k={clusters_estimate})"
        )

        plt.legend()

        plt.tight_layout()
        plt.savefig(f"figs/hierarchical_clustering_k{clusters_estimate}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
