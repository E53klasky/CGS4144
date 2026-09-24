import pandas as pd

rank_sum_results = pd.read_csv("Assignment-2/results/rank_sum.csv")
gprofiler_results = pd.read_csv("Assignment-2/results/enrichment_gprofiler_GO_BP.csv")
clusterprofiler_results = pd.read_csv("Assignment-2/results/GO_BP_enrichment_clusterProfiler.csv")

gprofiler_results = gprofiler_results.rename(columns={"native": "go_term"})
clusterprofiler_results = clusterprofiler_results.rename(columns={"ID": "go_term"})

rank_sum_results = rank_sum_results.rename(columns={"p_value": "p_value_rank_sum"})
gprofiler_results = gprofiler_results.rename(columns={"p_value": "p_value_gprofiler"})
clusterprofiler_results = clusterprofiler_results.rename(columns={"pvalue": "p_value_clusterprofiler"})

rank_sum_results = rank_sum_results[["go_term", "p_value_rank_sum"]]
gprofiler_results = gprofiler_results[["go_term", "p_value_gprofiler"]]
clusterprofiler_results = clusterprofiler_results[["go_term", "p_value_clusterprofiler"]]

combined_results = pd.merge(rank_sum_results, gprofiler_results, on="go_term", how="outer", suffixes=("_rank_sum", "_gprofiler"))
combined_results = pd.merge(combined_results, clusterprofiler_results, on="go_term", how="outer", suffixes=("", "_clusterprofiler"))
combined_results["methods"] = combined_results[["p_value_rank_sum", "p_value_gprofiler", "p_value_clusterprofiler"]].notnull().sum(axis=1)
combined_results["significant_methods"] = ((combined_results["p_value_rank_sum"] < 0.05).astype(int) +
                                          (combined_results["p_value_gprofiler"] < 0.05).astype(int) +
                                          (combined_results["p_value_clusterprofiler"] < 0.05).astype(int))

combined_results = combined_results.sort_values(["methods", "significant_methods"], ascending=False)

combined_results.to_csv("Assignment-2/results/combined_enrichment.csv", index=False)


combined_top10 = pd.read_csv("Assignment-2/results/combined_enrichment.csv").head(10)
combined_top10.to_csv("Assignment-2/results/top_10_combined_enrichment.csv", index=False)
