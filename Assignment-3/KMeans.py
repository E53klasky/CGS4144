import reader
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2_contingency


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

    k = 2

    gene_counts = [10, 100, 1000, 10000]

    results = []
    for n_genes in gene_counts:

        top_genes = gene_variance.nlargest(n_genes).index
        expression_subset = expression.loc[top_genes]

        X = expression_subset.T.values

        model = KMeans(n_clusters=k, random_state=42, n_init=20)

        clusters = model.fit_predict(X)

        contingency_table = pd.crosstab(metadata["refinebio_disease"], clusters)

        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        results.append((n_genes, chi2, p_value))

    p_values = [result[2] for result in results]

    adjusted_p_values = multipletests(p_values, method="fdr_bh")[1]

    print("\nK-MEANS RESULTS")
    print("k = 2\n")

    for result, adjusted_p in zip(results, adjusted_p_values):
        n_genes, chi2, p_value = result

        print(f"Genes: {n_genes}")
        print(f"Chi-Squared: {chi2}")
        print(f"P-Value: {p_value}")
        print(f"Adjusted P-Value: {adjusted_p}")
        print()


if __name__ == "__main__":
    main()
