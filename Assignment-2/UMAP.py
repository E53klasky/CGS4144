import reader
import torch
import matplotlib.pyplot as plt
import umap


def main():
    expression, metadata = reader.load_data()

    groups = ["tb subjects", "healthy controls"]

    metadata = metadata[metadata["refinebio_disease"].isin(groups)].copy()

    metadata = metadata.set_index("refinebio_accession_code")
    sample_ids = [
        sample
        for sample in expression.columns
        if sample in metadata.index
    ]

    expression = expression[sample_ids]
    metadata = metadata.loc[sample_ids]

    print("Samples per group:")
    print(metadata["refinebio_disease"].value_counts())

    print("Expression shape (genes x samples):", expression.shape)

    device = "cpu" 

    X = torch.tensor(
        expression.values,
        dtype=torch.float32,
        device=device,
    )

    X = torch.log2(X + 1)

    X = X.T

    print("UMAP input shape (samples x genes):", X.shape)

    X_np = X.detach().cpu().numpy()

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=67,
    )

    X_umap = reducer.fit_transform(X_np)

    disease = metadata["refinebio_disease"].to_numpy()

    tb_mask = disease == "tb subjects"
    healthy_mask = disease == "healthy controls"

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_umap[tb_mask, 0],
        X_umap[tb_mask, 1],
        label="TB Subjects",
        alpha=0.7,
    )

    plt.scatter(
        X_umap[healthy_mask, 0],
        X_umap[healthy_mask, 1],
        label="Healthy Controls",
        alpha=0.7,
    )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    plt.title("UMAP of SRP092402 RNA-seq Expression")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "figs/UMAP.png",
        dpi=300,
    )



if __name__ == "__main__":
    main()
