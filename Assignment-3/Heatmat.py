import reader
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans


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

    print("Heatmap expression shape:", expression.shape)

    X = expression.T.values

    k = 2

    model = KMeans(n_clusters=k, random_state=42, n_init=20)

    clusters = model.fit_predict(X)

    annotation = pd.DataFrame(
        {
            "K-means": [f"Cluster {x}" for x in clusters],
            "Group": metadata["refinebio_disease"].values,
        },
        index=sample_ids,
    )

    cluster_palette = sns.color_palette("Set2", n_colors=k)

    cluster_colors = {f"Cluster {i}": cluster_palette[i] for i in range(k)}

    group_palette = sns.color_palette("Set1", n_colors=len(groups))

    group_colors = {group: group_palette[i] for i, group in enumerate(groups)}

    column_colors = pd.DataFrame(
        {
            "K-means": annotation["K-means"].map(cluster_colors),
            "Group": annotation["Group"].map(group_colors),
        },
        index=sample_ids,
    )

    Path("figs").mkdir(exist_ok=True)

    heatmap = sns.clustermap(
        expression,
        col_colors=column_colors,
        row_cluster=True,
        col_cluster=True,
        cmap="vlag",
        yticklabels=False,
        xticklabels=False,
        figsize=(14, 12),
    )

    heatmap.ax_heatmap.set_xlabel("Samples")
    heatmap.ax_heatmap.set_ylabel("5,000 Most Variable Genes")

    heatmap.fig.suptitle("Gene Expression Heatmap with K-means Clusters", y=1.08)

    for label, color in cluster_colors.items():
        heatmap.ax_col_dendrogram.bar(
            0, 0, color=color, label=f"K-means: {label}", linewidth=0
        )

    for label, color in group_colors.items():
        heatmap.ax_col_dendrogram.bar(
            0, 0, color=color, label=label.title(), linewidth=0
        )

    heatmap.ax_col_dendrogram.legend(
        loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.35)
    )

    plt.savefig("figs/heatmap_5000_genes.png", dpi=300, bbox_inches="tight")

    plt.close()

    print("Saved: figs/heatmap_5000_genes.png")


if __name__ == "__main__":
    main()
