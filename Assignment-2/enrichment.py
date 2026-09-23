from pathlib import Path

import pandas as pd
from gprofiler import GProfiler


def main():

    Path("results").mkdir(exist_ok=True)


    de_file = Path("results/differential_expression_all.csv")

    results = pd.read_csv(de_file)

    # Use statistically significant differentially expressed genes
    significant = results[
        results["adjusted_p_value"] < 0.05
    ].copy()

    print("Significant DE genes:", len(significant))

    # The DE table currently contains ENSG identifiers.
    # g:Profiler accepts human Ensembl gene IDs directly.
    genes = (
        significant["gene"]
        .dropna()
        .astype(str)
        .str.split(".")
        .str[0]
        .drop_duplicates()
        .tolist()
    )

    print("Unique genes submitted:", len(genes))


    gp = GProfiler(
        return_dataframe=True,
    )

    enrichment = gp.profile(
        organism="hsapiens",
        query=genes,

        sources=["GO:BP"],

        significance_threshold_method="fdr",

        user_threshold=0.05,
    )

    print("Significantly enriched GO terms:", len(enrichment))


    output_file = Path(
        "results/enrichment_gprofiler_GO_BP.csv"
    )

    enrichment.to_csv(
        output_file,
        index=False,
    )

    print("Saved:", output_file)

    # Show some of the strongest results
    if not enrichment.empty:

        columns = [
            column
            for column in [
                "native",
                "name",
                "p_value",
                "intersection_size",
                "term_size",
            ]
            if column in enrichment.columns
        ]

        print("\nTop 10 enriched terms:")
        print(
            enrichment[columns]
            .sort_values("p_value")
            .head(10)
            .to_string(index=False)
        )

    else:
        print("No significant GO:BP terms were found.")


if __name__ == "__main__":
    main()
