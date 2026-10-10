import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

CLASS_ORDER = ["negative", "neutral", "positive"]


def evaluate(y_true, y_pred, model_name, results_dir, split_name="test"):
    """Score any model the same way: print the report, save metrics as JSON and the confusion matrix as PNG."""
    results_dir = Path(results_dir)
    metrics = {
        "model": model_name,
        "split": split_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "per_class": classification_report(y_true, y_pred, labels=CLASS_ORDER,
                                           output_dict=True, zero_division=0),
    }

    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics" / f"{model_name}_{split_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_ORDER).plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} ({split_name} set)")
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "figures" / f"cm_{model_name}_{split_name}.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"{model_name} on {split_name}: accuracy {metrics['accuracy']} | Macro-F1 {metrics['macro_f1']}")
    print(classification_report(y_true, y_pred, labels=CLASS_ORDER, digits=3, zero_division=0))
    return metrics