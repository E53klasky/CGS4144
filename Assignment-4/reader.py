from pathlib import Path

import matplotlib.pyplot as plt
import mygene
import numpy as np
import pandas as pd


def load_data(data_dir="../SRP092402"):
    """
    Load the refine.bio expression matrix and sample metadata.

    This function keeps the same API used by PCA.py, t-SNE.py,
    UMAP.py, and DE.py.

    Returns:
        expression: genes x samples
        metadata: sample metadata
    """

    data_dir = Path(data_dir)

    expression_file = data_dir / "SRP092402.tsv"
    metadata_file = data_dir / "metadata_SRP092402.tsv"

    expression = pd.read_csv(
        expression_file,
        sep="\t",
        index_col=0,
    )

    metadata = pd.read_csv(
        metadata_file,
        sep="\t",
    )

    return expression, metadata


def convert_ensembl_to_gene_names(expression):

    mg = mygene.MyGeneInfo()

    ensembl_ids = expression.index.astype(str).str.split(".").str[0].tolist()

    results = mg.querymany(
        ensembl_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
    )

    mapping = {}

    for result in results:
        if "symbol" in result:
            mapping[result["query"]] = result["symbol"]

    gene_names = [mapping.get(gene_id, gene_id) for gene_id in ensembl_ids]

    converted = expression.copy()
    converted.index = gene_names
    converted.index.name = "Gene"

    print(
        "Ensembl IDs converted to gene names:",
        sum(gene_id in mapping for gene_id in ensembl_ids),
    )

    return converted


def expression_variation(expression):
    """
    Assignment Step 1c:
    Log-scale the expression data and examine
    per-gene variation with a density plot.

    refine.bio metadata:
        quantile_normalized = true
        scale_by = NONE

    Therefore log2(x + 1) is applied here.
    """

    Path("figs").mkdir(exist_ok=True)

    # Log-scale expression values
    log_expression = np.log2(expression + 1)

    # Median expression for each gene across all samples
    gene_medians = log_expression.median(axis=1)

    print("\nStep 1c")
    print("Expression matrix size:", expression.shape)
    print("Number of genes:", expression.shape[0])

    print("\nPer-gene median expression:")
    print(gene_medians.describe())

    # Density plot
    plt.figure(figsize=(8, 6))

    gene_medians.plot.density()

    plt.xlabel("Median Gene Expression (log2 scale)")
    plt.ylabel("Density")
    plt.title("Distribution of Per-Gene Median Expression")

    plt.tight_layout()

    plt.savefig(
        "figs/gene_expression_density.png",
        dpi=300,
    )

    plt.close()


def main():

    expression, metadata = load_data()

    print("Expression matrix:")
    print(expression.shape)
    print(expression.iloc[:5, :5])

    print("\nMetadata:")
    print(metadata.shape)
    print(metadata.head())

    print("\nMetadata columns:")
    print(metadata.columns.tolist())

    expression_gene_names = convert_ensembl_to_gene_names(expression)

    print("\nExpression matrix with gene names:")
    print(expression_gene_names.iloc[:5, :5])

    expression_variation(expression_gene_names)


if __name__ == "__main__":
    main()
