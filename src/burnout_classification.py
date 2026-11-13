"""Revised educational benchmark on synthetic employee burnout data.

Run from the repository root: python src/burnout_classification.py
"""
from pathlib import Path
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tech_mental_health_burnout.csv"
OUTPUT = ROOT / "results"
LABELS = ["Low", "Moderate", "High"]


def main():
    df = pd.read_csv(DATA)
    assert set(LABELS) == set(df["burnout_level"].unique())

    X = df.drop(columns=["burnout_level", "burnout_score"])
    y = df["burnout_level"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    models = {
        "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
            class_weight="balanced_subsample"
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
    }
    OUTPUT.mkdir(exist_ok=True)
    results = []
    for name, model in models.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        report = classification_report(
            y_test, pred, labels=LABELS, output_dict=True, zero_division=0
        )
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
            "high_recall": report["High"]["recall"],
            "high_precision": report["High"]["precision"],
        }
        results.append(row)
        pd.DataFrame(
            confusion_matrix(y_test, pred, labels=LABELS),
            index=LABELS,
            columns=LABELS,
        ).to_csv(
            OUTPUT / (
                "confusion_" + name.lower().replace(" ", "_").replace("-", "_") + ".csv"
            )
        )
        print(name, {k: round(v, 4) for k, v in row.items() if k != "model"})

    pd.DataFrame(results).to_csv(OUTPUT / "model_comparison.csv", index=False)
    (OUTPUT / "dataset_summary.json").write_text(json.dumps({
        "rows": len(df),
        "columns": len(df.columns),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features_exclude": ["burnout_level", "burnout_score"],
        "random_state": 42,
    }, indent=2))


if __name__ == "__main__":
    main()
