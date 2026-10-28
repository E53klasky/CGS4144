import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, "../Assignment-3")
import reader


def main():

    os.makedirs("out", exist_ok=True)

    expression, metadata = reader.load_data("../SRP092402")

    metadata = metadata[
        metadata["refinebio_disease"].isin(["tb subjects", "healthy controls"])
    ].copy()

    metadata = metadata.set_index("refinebio_accession_code")

    sample_ids = [sample for sample in expression.columns if sample in metadata.index]

    expression = expression[sample_ids]
    metadata = metadata.loc[sample_ids]

    # Logistic regression gene signature
    coefficients = pd.read_csv("CGS4144/Assignment-4/results/disease_5000_genes.csv")

    top_genes = coefficients.head(50)["gene"]

    coefficients_rf = pd.read_csv("CGS4144/Assignment-4/RandForest_5000_genes.csv")
    rf_top = coefficients_rf.head(50)["Gene"]
    
    top_genes_combined = pd.Index(top_genes.tolist()+rf_top.tolist()).unique()

    heatmap_data = expression.loc[top_genes]

    heatmap_data = np.log2(heatmap_data + 1)

    heatmap_data = heatmap_data.sub(
        heatmap_data.mean(axis=1),
        axis=0,
    )

    heatmap_data = heatmap_data.div(
        heatmap_data.std(axis=1).replace(0, np.nan),
        axis=0,
    )

    heatmap_data = heatmap_data.fillna(0)

    disease = metadata["refinebio_disease"]

    group_colors = {
        "healthy controls": "royalblue",
        "tb subjects": "firebrick",
    }

    sample_colors = disease.map(group_colors)

    # Heatmap with row and column dendrograms
    g = sns.clustermap(
        heatmap_data,
        row_cluster=True,
        col_cluster=True,
        col_colors=sample_colors,
        cmap="vlag",
        center=0,
        xticklabels=False,
        yticklabels=True,
        figsize=(16, 12),
        cbar_kws={"label": "Gene expression (Z-score)"},
    )

    g.ax_heatmap.set_xlabel("Samples")
    g.ax_heatmap.set_ylabel("Genes")

    from matplotlib.patches import Patch

    legend = [
        Patch(color="royalblue", label="Healthy Controls"),
        Patch(color="firebrick", label="TB Subjects"),
    ]

    g.ax_col_dendrogram.legend(
        handles=legend,
        title="Sample Group",
        loc="upper right",
    )

    g.fig.suptitle(
        "Heatmap: TB vs Healthy",
        y=1.02,
    )

    g.savefig(
        "CGS4144/Assignment-4/heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


if __name__ == "__main__":
    main()
