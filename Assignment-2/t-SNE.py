import reader
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main():
    expression, metadata = reader.load_data()

    groups = ["tb subjects", "healthy controls"]

    metadata = metadata[metadata["refinebio_disease"].isin(groups)].copy()
    metadata = metadata.set_index("refinebio_accession_code")
    sample_ids = [sample for sample in expression.columns if sample in metadata.index]

    expression = expression[sample_ids]
    metadata = metadata.loc[sample_ids]

    print("Samples per group:")
    print(metadata["refinebio_disease"].value_counts())

    print("Expression shape (genes x samples):", expression.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.tensor(expression.values, dtype=torch.float32, device=device)
    X = torch.log2(X + 1)
    X = X.T

    print("t-SNE input shape (samples x genes):", X.shape)

    X_np_array = X.detach().cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=20, random_state=67)

    X_tsne = tsne.fit_transform(X_np_array)

    disease = metadata["refinebio_disease"].to_numpy()

    tb_mask = disease == "tb subjects"
    healthy_mask = disease == "healthy controls"

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    plt.scatter(
        X_tsne[tb_mask, 0],
        X_tsne[tb_mask, 1],
        label="TB Subjects",
        alpha=0.7,
    )

    plt.scatter(
        X_tsne[healthy_mask, 0],
        X_tsne[healthy_mask, 1],
        label="Healthy Controls",
        alpha=0.7,
    )

    plt.title("t-SNE plot of SRP092402 RNA-seq Expression")
    plt.legend()

    plt.tight_layout()
    plt.savefig("figs/t-SNE.png", dpi=300)
    # plt.show()

    print("END OF PROGRAM")


if __name__ == "__main__":
    main()
