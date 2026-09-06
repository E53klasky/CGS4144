from pathlib import Path
import pandas as pd


def load_data(data_dir="../SRP092402"):
    data_dir = Path(data_dir)

    expression_file = data_dir / "SRP092402.tsv"
    metadata_file = data_dir / "metadata_SRP092402.tsv"

    # Expression matrix:
    #   rows    = genes
    #   columns = samples
    expression = pd.read_csv(
        expression_file,
        sep="\t",
        index_col=0,
    )

    # Sample metadata
    metadata = pd.read_csv(
        metadata_file,
        sep="\t",
    )

    return expression, metadata


if __name__ == "__main__":
    expression, metadata = load_data()

    print("Expression matrix:")
    print(expression.shape)
    print(expression.iloc[:5, :5])

    print("\nMetadata:")
    print(metadata.shape)
    print(metadata.head())

    print("\nMetadata columns:")
    print(metadata.columns.tolist())
