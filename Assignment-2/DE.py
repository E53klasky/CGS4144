import reader
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from PyComplexHeatmap import (
    ClusterMapPlotter,
    HeatmapAnnotation,
    anno_simple,
)


def main():

    Path("figs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    expression, metadata = reader.load_data()

    groups = ["tb subjects", "healthy controls"]

    metadata = metadata[metadata["refinebio_disease"].isin(groups)].copy()
    metadata = metadata.set_index("refinebio_accession_code")

    sample_ids = [sample for sample in expression.columns if sample in metadata.index]

    expression = expression[sample_ids]
    metadata = metadata.loc[sample_ids]

    print("Samples per group:")
    print(metadata["refinebio_disease"].value_counts())

    log_expression = np.log2(expression.to_numpy() + 1)

    disease = metadata["refinebio_disease"].to_numpy()

    tb_mask = disease == "tb subjects"
    healthy_mask = disease == "healthy controls"

    tb = log_expression[:, tb_mask]
    healthy = log_expression[:, healthy_mask]

    print("TB matrix:", tb.shape)
    print("Healthy matrix:", healthy.shape)

    tb_mean = tb.mean(axis=1)
    healthy_mean = healthy.mean(axis=1)

    log2_fc = tb_mean - healthy_mean

    # Welch's t-test for each gene
    _, p_values = ttest_ind(
        tb,
        healthy,
        axis=1,
        equal_var=False,
        nan_policy="omit",
    )

    # Benjamini-Hochberg FDR correction
    _, adjusted_p, _, _ = multipletests(
        p_values,
        alpha=0.05,
        method="fdr_bh",
    )

    results = pd.DataFrame(
        {
            "gene": expression.index,
            "tb_mean": tb_mean,
            "healthy_mean": healthy_mean,
            "log2_fold_change": log2_fc,
            "p_value": p_values,
            "adjusted_p_value": adjusted_p,
        }
    )

    # Significant genes
    results["significant"] = results["adjusted_p_value"] < 0.05

    print(
        "Genes with adjusted p < 0.05:",
        (results["adjusted_p_value"] < 0.05).sum(),
    )

    print(
        "Genes with |log2FC| >= 1:",
        (np.abs(results["log2_fold_change"]) >= 1).sum(),
    )

    print(
        "Genes satisfying both:",
        (
            (results["adjusted_p_value"] < 0.05)
            & (np.abs(results["log2_fold_change"]) >= 1)
        ).sum(),
    )

    results = results.sort_values("adjusted_p_value")

    results.to_csv(
        "results/differential_expression_all.csv",
        index=False,
    )

    results.head(50).to_csv(
        "results/top_50_differential_genes.csv",
        index=False,
    )

    print("\nSignificant genes:")
    print(results["significant"].sum())

    print("\nTop 10 genes:")
    print(
        results[
            [
                "gene",
                "log2_fold_change",
                "adjusted_p_value",
            ]
        ].head(10)
    )

    make_volcano_plot(results)

    make_heatmap(
        expression,
        metadata,
        results,
    )


def make_volcano_plot(results):

    x = results["log2_fold_change"].to_numpy()
    p = results["adjusted_p_value"].clip(lower=1e-300).to_numpy()

    y = -np.log10(p)

    significant = results["significant"].to_numpy()

    plt.figure(figsize=(8, 6))

    # Non-significant genes
    plt.scatter(
        x[~significant],
        y[~significant],
        alpha=0.4,
        label="Not significant",
    )

    # Significant genes
    plt.scatter(
        x[significant],
        y[significant],
        alpha=0.7,
        label="Significant",
    )

    # FDR = 0.05 threshold
    plt.axhline(
        -np.log10(0.05),
        linestyle="--",
    )

    plt.xlabel("log2 Fold Change (TB / Healthy)")
    plt.ylabel("-log10 Adjusted p-value")

    plt.title("Differential Expression: TB vs Healthy Controls")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "figs/volcano_plot.png",
        dpi=300,
    )

    plt.close()


def make_heatmap(expression, metadata, results):

    significant_genes = results.loc[
        results["significant"],
        "gene",
    ]

    heatmap_data = expression.loc[significant_genes]

    # log2 transform
    heatmap_data = np.log2(heatmap_data + 1)

    # z-score each gene across samples
    heatmap_data = heatmap_data.sub(
        heatmap_data.mean(axis=1),
        axis=0,
    )

    heatmap_data = heatmap_data.div(
        heatmap_data.std(axis=1),
        axis=0,
    )

    disease = metadata["refinebio_disease"]

    # Healthy first, then TB
    group_order = {
        "healthy controls": 0,
        "tb subjects": 1,
    }

    order = disease.map(group_order).sort_values().index

    heatmap_data = heatmap_data[order]
    disease = disease.loc[order]

    sample_groups = pd.Series(
        [
            "Healthy Controls" if group == "healthy controls" else "TB Subjects"
            for group in disease
        ],
        index=disease.index,
        name="Sample Group",
    )

    col_annotation = HeatmapAnnotation(
        Sample_Group=anno_simple(
            sample_groups,
            colors={
                "Healthy Controls": "royalblue",
                "TB Subjects": "firebrick",
            },
        ),
        axis=1,
    )

    plt.figure(figsize=(15, 10))

    ClusterMapPlotter(
        data=heatmap_data,
        top_annotation=col_annotation,
        col_cluster=False,
        row_cluster=True,
        show_rownames=True,
        show_colnames=False,
        cmap="viridis",
        legend=True,
    )

    plt.savefig(
        "figs/DE_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


if __name__ == "__main__":
    main()
