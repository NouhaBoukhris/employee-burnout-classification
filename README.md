# Employee Burnout Classification | Machine Learning Benchmark

**Python · pandas · scikit-learn · HR Analytics · Classification**

A collaborative Tunis Business School data-mining project examining burnout-level classification using **150,000 synthetic technology-sector employee records**. This portfolio edition provides a revised, leakage-aware benchmark and explains limitations of the original coursework.

## Research question

Can workplace, behavioral, and self-reported psychological indicators classify employee burnout levels (**Low**, **Moderate**, **High**), and what model trade-offs matter when the High class is rare?

## Dataset and limitations

- **Source:** [Kaggle — Employee Mental Health and Burnout Dataset](https://www.kaggle.com/datasets/suhanigupta04/employee-mental-health-and-burnout-dataset).
- **Synthetic records:** 150,000; 25 columns, including the target and a target-related burnout score.
- **Class distribution:** Low: 131,620; Moderate: 18,315; High: **65 (0.043%)**.
- **Not clinical evidence:** Synthetic labels cannot validate mental health diagnoses, interventions, or real-world model deployment. Any HR application would require independent validation, privacy, consent, and fairness safeguards.

## Methodology: revised benchmark

1. Exclude `burnout_level` (target) and `burnout_score` (closely related composite) from predictors.
2. Make a stratified 80/20 train/test split (`random_state=42`).
3. Fit numeric imputation and scaling, plus categorical imputation and one-hot encoding, **using training data only**, within scikit-learn pipelines.
4. Evaluate a majority-class baseline, Logistic Regression, Decision Tree, Random Forest, KNN, and Gaussian Naive Bayes.
5. Compare accuracy with balanced accuracy, macro-F1, High-class recall and precision, and confusion matrices.

Class weighting is enabled for Logistic Regression, Decision Tree, and Random Forest. The revised benchmark differs from the original academic results; do not treat their metrics as interchangeable.

## Revised benchmark results

| Model | Accuracy | Balanced accuracy | Macro-F1 | High recall | High precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority-class baseline | 87.75% | 33.33% | 0.312 | 0.00% | 0.00% |
| Logistic Regression | 88.15% | 84.41% | 0.570 | 76.92% | 7.41% |
| Decision Tree | 83.12% | 72.03% | 0.497 | 53.85% | 3.07% |
| Random Forest | **92.41%** | 49.36% | 0.524 | 0.00% | 0.00% |
| KNN | 91.23% | 48.00% | 0.505 | 0.00% | 0.00% |
| Naive Bayes | 89.57% | 74.02% | 0.589 | 53.85% | 11.11% |

**Interpretation:** Random Forest has the best overall accuracy but identifies **none** of the 13 High cases in the test set. Logistic Regression finds 10 of 13, at the cost of many false positives. Results for High are unstable due to the tiny test count. This project is an educational comparison, not a deployable screening system.

## Run locally

Install dependencies and run the benchmark from the repository root:

```bash
pip install -r requirements.txt
python src/burnout_classification.py
```

The script reads `data/tech_mental_health_burnout.csv` (included in the [downloadable portfolio archive](https://github.com/NouhaBoukhris/employee-burnout-classification); upload to `data/` if absent) and writes performance metrics and confusion matrices to `results/`.

## Repository contents

- [`src/burnout_classification.py`](src/burnout_classification.py) — revised modeling code.
- [`results/`](results/) — benchmark scores, summary and confusion matrices.
- `data/tech_mental_health_burnout.csv` — Kaggle synthetic input (upload separately; check redistribution terms).
- `docs/original_coursework.py` — original academic script (upload separately if needed).
- `docs/original_report.docx` — updated original report with corrected Drive link (upload separately).
- `docs/presentation.pdf` — original 21-slide presentation (upload separately).

**Note:** The original academic report and presentation retain the original model results and class-label narrative; the revised benchmark above is deliberately separate.

## Academic context

**Data Mining (BA360)** · Tunis Business School · 2025–2026. Team: Alaa Azouzi, Sarra Alioua, Nouha Boukhris, and Mariem Chammem. The project was collaborative; no individual contribution is implied by the repository ownership.

## Further improvements

Validate on an independently sourced real-world dataset with appropriate consent; use repeated stratified evaluation, precision-recall curves, probability calibration and fairness audits before any real deployment.
