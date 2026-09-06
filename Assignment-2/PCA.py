import reader  # I/O for data
import torch
import matplotlib.pyplot as plt


def main():

    expression, metadata = reader.load_data()

    # Keep only TB subjects and healthy controls
    groups = ["tb subjects", "healthy controls"]

    metadata = metadata[metadata["refinebio_disease"].isin(groups)].copy()

    # Make metadata order match the expression matrix
    metadata = metadata.set_index("refinebio_accession_code")
    sample_ids = [sample for sample in expression.columns if sample in metadata.index]

    expression = expression[sample_ids]
    metadata = metadata.loc[sample_ids]

    print("Samples per group:")
    print(metadata["refinebio_disease"].value_counts())

    print("Expression shape (genes x samples):", expression.shape)

    X = torch.tensor(expression.values, dtype=torch.float32)
    X = torch.log2(X + 1)
    X = X.T

    print("PCA input shape (samples x genes):", X.shape)

    X = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    X_pca = X @ Vh[:2].T
    variance = S**2 / (X.shape[0] - 1)
    explained_variance_ratio = variance / variance.sum()

    print("PC1 explained variance:", explained_variance_ratio[0].item())
    print("PC2 explained variance:", explained_variance_ratio[1].item())

    # For plotting
    X_pca = X_pca.cpu().numpy()

    # Plot
    plt.figure(figsize=(8, 6))

    disease = metadata["refinebio_disease"].to_numpy()

    tb_mask = disease == "tb subjects"
    healthy_mask = disease == "healthy controls"

    plt.scatter(
        X_pca[tb_mask, 0],
        X_pca[tb_mask, 1],
        label="TB Subjects",
        alpha=0.7,
    )

    plt.scatter(
        X_pca[healthy_mask, 0],
        X_pca[healthy_mask, 1],
        label="Healthy Controls",
        alpha=0.7,
    )

    plt.xlabel(f"PC1 ({explained_variance_ratio[0].item() * 100:.1f}% variance)")

    plt.ylabel(f"PC2 ({explained_variance_ratio[1].item() * 100:.1f}% variance)")

    plt.title("PCA of SRP092402 RNA-seq Expression")
    plt.legend()

    plt.tight_layout()
    plt.savefig("figs/PCA.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    main()
