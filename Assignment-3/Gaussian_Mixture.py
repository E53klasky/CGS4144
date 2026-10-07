import reader
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency


def main():
    expr, meta = reader.load_data()
    meta = meta[
        meta["refinebio_disease"].isin(["tb subjects", "healthy controls"])
    ].copy()

    meta = meta.set_index("refinebio_accession_code")
    ids = expr.columns.intersection(meta.index)
    expr = expr[ids]
    meta = meta.loc[ids]

    expr = np.log2(expr + 1)
    var = expr.var(axis=1)
    og_expr = expr.copy()
    most_var = var.nlargest(5000).index
    expr_1 = og_expr.loc[most_var]

    x_values = expr_1.T
    x_values_scaled = StandardScaler().fit_transform(x_values)

    for i in range(2, 8):
        model = GaussianMixture(n_components=i, random_state=42)
        clusters = model.fit_predict(x_values_scaled)
        val = silhouette_score(x_values_scaled, clusters)
        print("Current K:", i)
        print("Score (Silhouette):", val)
        print("Num in Clusters: ")
        print(pd.Series(clusters).value_counts().sort_index())

        tb_table = pd.crosstab(clusters, meta["refinebio_disease"]).values
        chi,p_val,temp,temp2 = chi2_contingency(tb_table)
        print("TB vs Healthy Chi-Squared Tests: ")
        print(tb_table)
        print("Chi-Squared Value:", chi)
        print("P-Value:", p_val)

    num_genes = [10, 100, 1000, 10000]
    diff_genes_results = {}
    for num in num_genes:
        expr_subset = expr.loc[var.nlargest(num).index]
        x_values_subset = expr_subset.T
        x_subset_scaled = StandardScaler().fit_transform(x_values_subset)

        model_genes = GaussianMixture(n_components=3, random_state=42)
        clusters_genes = model_genes.fit_predict(x_subset_scaled)
        diff_genes_results[num] = clusters_genes
        print("Current Num Genes:", num)
        print(pd.Series(clusters_genes).value_counts().sort_index())
    gene_num_pairs = [
        (10, 100),
        (10, 1000),
        (10, 10000),
        (100, 1000),
        (100, 10000),
        (1000, 10000),
    ]

    for num1, num2 in gene_num_pairs:
        gen_table = pd.crosstab(diff_genes_results[num1], diff_genes_results[num2])
        chi2, p, temp, temp2 = chi2_contingency(gen_table)
        print("Gene count pair:", num1, num2)
        print("Chi-Squared Value:", chi2)
        print("P-Value:", p)


if __name__ == "__main__":
    main()
