from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import false_discovery_control, fisher_exact, hypergeom, spearmanr

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "results" / "section3"


def read_predictions(path, sample="sample", prediction="predicted"):
    frame = pd.read_csv(BASE / path).rename(
        columns={sample: "sample", prediction: "predicted"}
    )
    if frame["sample"].isna().any() or frame["sample"].duplicated().any():
        raise ValueError(f"{path}: missing or duplicate sample IDs")
    if not frame["predicted"].isin([0, 1]).all():
        raise ValueError(f"{path}: expected binary predictions")
    return frame.set_index("sample")["predicted"]


def align_clusters(predictions, reference_path, candidate_path, test_ids):
    reference = pd.read_csv(BASE / reference_path).set_index("sample")["cluster"]
    candidate_frame = pd.read_csv(BASE / candidate_path)
    if candidate_frame["sample"].duplicated().any():
        raise ValueError(f"{candidate_path}: duplicate sample IDs")
    candidate = candidate_frame.set_index("sample")["cluster"]
    if set(reference.index) != set(candidate.index):
        raise ValueError(
            f"{candidate_path}: cluster partitions must cover the same samples"
        )
    if set(reference.unique()) != {0, 1} or set(candidate.unique()) != {0, 1}:
        raise ValueError(
            "Cluster alignment currently requires two clusters encoded 0 and 1"
        )
    train = reference.index[~reference.index.isin(test_ids)]
    table = pd.crosstab(reference.loc[train], candidate.loc[train]).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0
    )
    if table.to_numpy().trace() == np.fliplr(table.to_numpy()).trace():
        raise ValueError("Cluster label alignment is tied; no unique mapping")
    rows, columns = linear_sum_assignment(-table.to_numpy())
    mapping = dict(zip(columns.tolist(), rows.tolist()))
    return predictions.map(mapping), mapping


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = {
        "LR_disease": ("results/disease_5000_predictions.csv", "sample", "predicted"),
        "LR_cluster": ("results/cluster_5000_predictions.csv", "sample", "predicted"),
        "RF_disease": (
            "RandForest_Predictions_5000_Genes.csv",
            "Sample",
            "Predicted Label",
        ),
        "RF_cluster": (
            "RandomForest_Predicted_Clusters.csv",
            "Sample",
            "Predicted Cluster",
        ),
        "NB_disease": ("out/nb_disease_5000_predictions.csv", "sample", "predicted"),
        "NB_cluster": ("out/nb_cluster_5000_predictions.csv", "sample", "predicted"),
    }
    expected = pd.Index(pd.read_csv(BASE / "results/test_sample_ids.csv")["sample"])
    if expected.has_duplicates:
        raise ValueError("Shared test IDs contain duplicates")
    audit, predictions, problems = {}, {}, []
    for name, args in sources.items():
        if not (BASE / args[0]).exists():
            audit[name] = {"file": args[0], "status": "missing"}
            problems.append(f"{name}: missing {args[0]}")
            continue
        values = read_predictions(*args)
        predictions[name] = values
        audit[name] = {
            "file": args[0],
            "samples": len(values),
            "predicted_0": int(values.eq(0).sum()),
            "predicted_1": int(values.eq(1).sum()),
            "shared_test_overlap": len(expected.intersection(values.index)),
        }
        if set(values.index) != set(expected):
            problems.append(f"{name}: test samples differ from the shared split")
    for path in ["results/rf_sample_clusters.csv", "out/nb_sample_clusters.csv"]:
        if not (BASE / path).exists():
            problems.append(f"Missing full cluster partition for alignment: {path}")
    audit["problems"] = problems
    (OUTPUT / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if problems:
        print("Section 3 is incomplete. See results/section3/input_audit.json.")
        for problem in problems:
            print("-", problem)
        return

    mappings = {}
    for model, partition in [
        ("RF", "results/rf_sample_clusters.csv"),
        ("NB", "out/nb_sample_clusters.csv"),
    ]:
        predictions[f"{model}_cluster"], mappings[model] = align_clusters(
            predictions[f"{model}_cluster"],
            "results/sample_labels.csv",
            partition,
            expected,
        )
    matrix = pd.DataFrame(
        {name: values.reindex(expected) for name, values in predictions.items()}
    )
    for target in ["disease", "cluster"]:
        columns = [f"{model}_{target}" for model in ["LR", "RF", "NB"]]
        matrix[f"{target}_votes_1"] = matrix[columns].sum(axis=1).astype(int)
        matrix[f"{target}_votes_0"] = 3 - matrix[f"{target}_votes_1"]
        matrix[f"{target}_stability"] = (
            matrix[[f"{target}_votes_0", f"{target}_votes_1"]].max(axis=1) / 3
        )
    matrix.index.name = "sample"
    matrix.to_csv(OUTPUT / "sample_votes.csv")
    disease_stable = matrix.disease_stability.eq(1)
    cluster_stable = matrix.cluster_stability.eq(1)
    table = pd.crosstab(disease_stable, cluster_stable).reindex(
        index=[False, True], columns=[False, True], fill_value=0
    )
    table.to_csv(OUTPUT / "stability_contingency.csv")
    if disease_stable.nunique() < 2 or cluster_stable.nunique() < 2:
        result = {
            "status": "Correlation is undefined because stability is constant",
            "cluster_mappings": mappings,
        }
    else:
        rho, asymptotic_p = spearmanr(
            matrix.disease_stability, matrix.cluster_stability
        )
        n = len(matrix)
        a, b = int(disease_stable.sum()), int(cluster_stable.sum())
        overlaps = np.arange(max(0, a + b - n), min(a, b) + 1)
        null_rho = (n * overlaps - a * b) / np.sqrt(a * (n - a) * b * (n - b))
        spearman_p = hypergeom.pmf(overlaps, n, a, b)[
            np.abs(null_rho) >= abs(rho) - 1e-12
        ].sum()
        odds, fisher_p = fisher_exact(table.to_numpy())
        # BH correction over the two explicitly reported association tests.
        corrected = false_discovery_control([spearman_p, fisher_p], method="bh")
        result = {
            "samples": len(matrix),
            "spearman_rho": float(rho),
            "spearman_p": float(spearman_p),
            "spearman_p_method": "exact permutation distribution of absolute rho",
            "spearman_asymptotic_p_reference_only": float(asymptotic_p),
            "spearman_BH_q": float(corrected[0]),
            "fisher_odds_ratio": float(odds) if np.isfinite(odds) else "infinite",
            "fisher_p": float(fisher_p),
            "fisher_BH_q": float(corrected[1]),
            "disease_unanimous": int(disease_stable.sum()),
            "cluster_unanimous": int(cluster_stable.sum()),
            "cluster_mappings": mappings,
        }
    (OUTPUT / "association.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
