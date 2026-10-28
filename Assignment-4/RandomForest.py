import reader
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import reader


expr,meta = reader.load_data("../SRP092402")
meta = meta[
    meta["refinebio_disease"].isin(["tb subjects", "healthy controls"])
]
meta = meta.set_index("refinebio_accession_code")
samples = expr.columns.intersection(meta.index)
x_axis = np.log2(expr[samples] + 1).T
y_axis = (meta.loc[samples, "refinebio_disease"] == "tb subjects").astype(int)

x_train, x_test, y_train, y_test = train_test_split(x_axis, y_axis, test_size=0.2, random_state=42)

diff_gene_nums = [10, 100, 1000,5000,10000]
for i in range(len(diff_gene_nums)):
    genes_var = x_train.var().nlargest(diff_gene_nums[i]).index

    train=x_train[genes_var]
    test=x_test[genes_var]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train, y_train)
    if diff_gene_nums[i] == 5000:
        rf_genes = pd.DataFrame({"Gene": genes_var, "Importance": model.feature_importances_})
        rf_genes = rf_genes.sort_values(by="Importance", ascending=False)
        rf_genes.to_csv("CGS4144/Assignment-4/RandForest_5000_genes.csv", index=False)
    print("Number of Genes:", diff_gene_nums[i])
    predict = model.predict(test)

    #1 = TB, 0 = Healthy
    if diff_gene_nums[i] == 5000:
        results = pd.DataFrame({"Sample": test.index, "Predicted Label": predict})
        results.to_csv("CGS4144/Assignment-4/RandForest_Predictions_5000_Genes.csv", index=False)
    print("Accuracy:", accuracy_score(y_test, predict))
    probab = model.predict_proba(test)[:, 1] 
    print("AUC-ROC:", roc_auc_score(y_test, probab))
    print("")

print("")
print("================================")
print("")

genes_var = x_train.var().nlargest(5000).index
x=x_axis[genes_var]
scaledX=StandardScaler().fit_transform(x_axis[genes_var])

gauss_model = GaussianMixture(n_components=3, random_state=42)
clusters = gauss_model.fit_predict(scaledX)
x_gtrain, x_gtest, y_gtrain, y_gtest = train_test_split(scaledX, clusters, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_gtrain, y_gtrain)
predict_rf_clus = rf_model.predict(x_gtest)

print("Accuracy - Clusters: ", accuracy_score(y_gtest, predict_rf_clus))

#0 = Cluster 0, 1 = Cluster 1, 2 = Cluster 2
results_clus = pd.DataFrame({"Sample": x_axis.index, "Predicted Cluster": clusters})
results_clus.to_csv("CGS4144/Assignment-4/RandomForest_Predicted_Clusters.csv", index=False)