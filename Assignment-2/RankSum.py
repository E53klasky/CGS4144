import pandas as pd
from scipy.stats import ranksums


deg = pd.read_csv('Assignment-2/results/differential_expression_all.csv')

gene_ontology = pd.read_csv("Assignment-2/mart_export.txt", sep=None, engine="python")

gene_ontology.columns = ["gene", "go_term", "go_name"]
gene_ontology = gene_ontology.dropna(subset=["go_term"])
gene_ontology = gene_ontology[gene_ontology["gene"].isin(deg["gene"])]
gene_ontology = gene_ontology.drop_duplicates(subset=["gene", "go_term"])

results = []

for go_term, group in gene_ontology.groupby("go_term"):
    term_genes = set(group["gene"])
    in_term = deg[deg["gene"].isin(term_genes)]["log2_fold_change"]
    not_in_term = deg[~deg["gene"].isin(term_genes)]["log2_fold_change"]
    if len(in_term)>=5:
        stat, p_value = ranksums(in_term, not_in_term)
        results.append({"go_term": go_term, "number_of_genes": len(in_term), "statistic": stat, "p_value": p_value})
results=pd.DataFrame(results)
results.to_csv("Assignment-2/results/rank_sum.csv", index=False)