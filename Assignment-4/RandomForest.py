import argparse

import reader
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument(
    "--reuse-clusters", action="store_true", help="Reuse results/rf_sample_clusters.csv"
)
args = parser.parse_args()

expr, meta = reader.load_data("../SRP092402")
meta = meta[meta["refinebio_disease"].isin(["tb subjects", "healthy controls"])]
meta = meta.set_index("refinebio_accession_code")
samples = expr.columns.intersection(meta.index)
x_axis = np.log2(expr[samples] + 1).T
y_axis = (meta.loc[samples, "refinebio_disease"] == "tb subjects").astype(int)

# Reuse the held-out IDs shared by logistic regression and Naive Bayes.
test_ids = pd.read_csv("results/test_sample_ids.csv")["sample"].tolist()
if len(test_ids) != len(set(test_ids)) or not set(test_ids).issubset(x_axis.index):
    raise ValueError("Test IDs must be unique and present in the expression matrix")
train_ids = x_axis.index[~x_axis.index.isin(test_ids)]
x_train, x_test = x_axis.loc[train_ids], x_axis.loc[test_ids]
y_train, y_test = y_axis.loc[train_ids], y_axis.loc[test_ids]

summaries = []
diff_gene_nums = [10, 100, 1000, 5000, 10000]
for i in range(len(diff_gene_nums)):
    genes_var = x_train.var().nlargest(diff_gene_nums[i]).index

    train = x_train[genes_var]
    test = x_test[genes_var]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train, y_train)
    if diff_gene_nums[i] == 5000:
        rf_genes = pd.DataFrame(
            {"Gene": genes_var, "Importance": model.feature_importances_}
        )
        rf_genes = rf_genes.sort_values(by="Importance", ascending=False)
        rf_genes.to_csv("RandForest_5000_genes.csv", index=False)
    print("Number of Genes:", diff_gene_nums[i])
    predict = model.predict(test)

    # 1 = TB, 0 = Healthy
    if diff_gene_nums[i] == 5000:
        results = pd.DataFrame(
            {
                "Sample": test.index,
                "actual": y_test.values,
                "Predicted Label": predict,
                "probability": model.predict_proba(test)[:, 1],
            }
        )
        results.to_csv("RandForest_Predictions_5000_Genes.csv", index=False)
    print("Accuracy:", accuracy_score(y_test, predict))
    probab = model.predict_proba(test)[:, 1]
    print("AUC-ROC:", roc_auc_score(y_test, probab))
    summaries.append(
        {
            "Target": "disease",
            "Genes": diff_gene_nums[i],
            "Accuracy": accuracy_score(y_test, predict),
            "AUC": roc_auc_score(y_test, probab),
        }
    )
    print("")

print("")
print("================================")
print("")

# Reproduce the full-data gene selection in Assignment 3/Gaussian_Mixture.py.
genes_var = x_axis.var().nlargest(5000).index
x = x_axis[genes_var]
if args.reuse_clusters:
    clusters = (
        pd.read_csv("results/rf_sample_clusters.csv")
        .set_index("sample")
        .loc[x.index, "cluster"]
        .to_numpy()
    )
else:
    scaledX = StandardScaler().fit_transform(x)
    gauss_model = GaussianMixture(n_components=2, random_state=42)
    clusters = gauss_model.fit_predict(scaledX)
# Save Tim's complete cluster partition for label alignment in Section 3.
cluster_labels = pd.Series(clusters, index=x.index, name="cluster")
pd.DataFrame({"sample": x.index, "cluster": clusters}).to_csv(
    "results/rf_sample_clusters.csv", index=False
)
y_gtrain, y_gtest = cluster_labels.loc[train_ids], cluster_labels.loc[test_ids]
for n_genes in diff_gene_nums:
    selected = x_train.var().nlargest(n_genes).index
    x_gtrain, x_gtest = x_train[selected], x_test[selected]
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_gtrain, y_gtrain)
    predicted = rf_model.predict(x_gtest)
    probabilities = rf_model.predict_proba(x_gtest)[:, 1]
    summaries.append(
        {
            "Target": "cluster",
            "Genes": n_genes,
            "Accuracy": accuracy_score(y_gtest, predicted),
            "AUC": roc_auc_score(y_gtest, probabilities),
        }
    )
    print("Cluster genes:", n_genes, summaries[-1])
    if n_genes == 5000:
        pd.DataFrame(
            {
                "Sample": x_gtest.index,
                "actual": y_gtest.values,
                "Predicted Cluster": predicted,
                "probability": probabilities,
            }
        ).to_csv("RandomForest_Predicted_Clusters.csv", index=False)
pd.DataFrame(summaries).to_csv("results/random_forest_model_summary.csv", index=False)
