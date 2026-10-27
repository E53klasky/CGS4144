import reader
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

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

    print("Number of Genes:", diff_gene_nums[i])
    predict = model.predict(test)

    #1 = TB, 0 = Healthy
    if diff_gene_nums[i] == 5000:
        results = pd.DataFrame({"Sample": test.index, "Predicted Label": predict})
        results.to_csv("RandomForest_Predictions_5000_Genes.csv", index=False)
    print("Accuracy:", accuracy_score(y_test, predict))
    probab = model.predict_proba(test)[:, 1] 
    print("AUC-ROC:", roc_auc_score(y_test, probab))
    print("")