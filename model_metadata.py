import json
import os

from constant import MODEL_DIR, RISK_DECISION_THRESHOLD


MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "stroke_risk_model.metadata.json")
REPORT_METADATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "metrics_report.json")


def save_model_metadata(metadata):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def load_model_metadata():
    if not os.path.exists(MODEL_METADATA_PATH):
        return _load_report_metadata()
    try:
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
        return _load_report_metadata()
    except Exception:
        return _load_report_metadata()


def _load_report_metadata():
    if not os.path.exists(REPORT_METADATA_PATH):
        return {}
    try:
        with open(REPORT_METADATA_PATH, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        dataset = report.get("dataset", {}) if isinstance(report, dict) else {}
        backend = dataset.get("backend")
        metadata = {
            "generated_at": report.get("generated_at"),
            "model_path": None,
            "backend": backend if backend in {"classical", "automorph"} else "classical",
            "risk_decision_threshold": report.get("deploy_threshold_metrics", {}).get(
                "threshold", float(RISK_DECISION_THRESHOLD)
            ),
            "dataset_path": dataset.get("path"),
            "total_samples": dataset.get("total_samples"),
            "label_0_normal": dataset.get("label_0_normal"),
            "label_1_high_risk": dataset.get("label_1_high_risk"),
            "best_hyperparameters": report.get("best_hyperparameters"),
        }
        return metadata
    except Exception:
        return {}


def get_model_backend(default="classical"):
    metadata = load_model_metadata()
    backend = metadata.get("backend")
    if backend in {"classical", "automorph"}:
        return backend
    return default


def get_model_threshold(default=None):
    metadata = load_model_metadata()
    threshold = metadata.get("risk_decision_threshold")
    if isinstance(threshold, (int, float)):
        return float(threshold)
    if default is not None:
        return float(default)
    return float(RISK_DECISION_THRESHOLD)