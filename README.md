# Telco Customer Churn Prediction

## Project overview
This college project explores customer churn in the IBM/WA Telco Customer Churn dataset and compares five supervised classification approaches:

- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Soft Voting Classifier

The target is `Churn` (`Yes`/`No`), encoded as a binary label. The goal is to identify customers who may leave the service, while comparing model performance using accuracy, precision, recall, F1-score, cross-validation, and precision–recall curves.

## Dataset
The script expects the file `WA_Fn-UseC_-Telco-Customer-Churn.csv`. The original script uses a Windows-specific path; update it to a path on your own machine, for example:

```python
dataset = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

The script reports **7,043 rows and 21 columns** before feature engineering. `TotalCharges` is converted to numeric; values that cannot be converted are filled with `0`. `customerID` is removed before modelling.

> **Data note:** filling unparseable `TotalCharges` with zero is a practical choice in the supplied script, but it can affect the analysis. For a more rigorous experiment, consider treating these as missing and documenting the imputation strategy.

## Workflow
1. Load the CSV and inspect its shape, types, missing values, and duplicate rows.
2. Convert `TotalCharges` to numeric.
3. Calculate descriptive statistics for `SeniorCitizen`, `tenure`, `MonthlyCharges`, and `TotalCharges`.
4. Encode categorical fields with one-hot, ordinal, and label encoding.
5. Split into training and test sets (80/20, `random_state=42`).
6. Standardize features using `StandardScaler` fitted on the training data.
7. Fit and evaluate four individual classifiers and a soft-voting ensemble.
8. Use cross-validation and `GridSearchCV` for parameter exploration.
9. Compare model scores and inspect the supplied charts.

## Reported results
The values below are the results recorded in the supplied Python script and charts. Accuracy is the percentage of test rows classified correctly.

| Model | Test accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 81.83% | 68.00% | 59.25% | 63.32% |
| Decision Tree | 77.57% | 64.02% | 32.44% | 43.06% |
| SVM | 81.76% | 68.01% | 58.71% | 63.02% |
| KNN | 79.91% | 63.64% | 56.30% | 59.74% |
| Soft Voting Classifier | 82.04% | 70.13% | 56.03% | 62.30% |

### Key findings / data story
- The **Soft Voting Classifier** has the highest reported test accuracy (82.04%), but its advantage over Logistic Regression (81.83%) is small.
- **Logistic Regression** has a slightly higher recall and F1-score than the ensemble in the reported results. If finding as many churners as possible matters, recall deserves particular attention.
- **Decision Tree** has the lowest test accuracy and notably lower recall (32.44%), meaning it misses a larger share of actual churners at the reported operating point.
- The **SVM** has performance close to Logistic Regression. The precision–recall curve legends report AP ≈ 0.68 for Logistic Regression and AP ≈ 0.67 for SVM; Decision Tree reports AP ≈ 0.48 and KNN ≈ 0.60.
- The numeric-feature heatmap shows a strong positive association between `tenure` and `TotalCharges` (0.83), and between `MonthlyCharges` and `TotalCharges` (0.65). Correlation is not proof of causation.
- Accuracy alone does not tell the whole story. Since churn is the positive class of interest, recall, precision, F1-score, and the cost of false negatives should be discussed alongside accuracy.

## How to run
### Requirements
Python 3.9+ is a reasonable starting point. Install the libraries used by the script:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn reportlab
```

### Run
1. Put `Project 2.py` and the CSV dataset in the same folder.
2. Change the `pd.read_csv(...)` path in the script to the CSV's local path.
3. Run:

```bash
python "Project 2.py"
```

The script includes chart code that is commented out in places. Uncomment the relevant plotting blocks if you want to regenerate some charts. The supplied charts are included with the project documentation PDF.

## Important reproducibility and methodology notes
- The script's original `read_csv` path is machine-specific and must be changed before running elsewhere.
- The code one-hot encodes categorical columns and concatenates the encoded columns to the original data, then excludes the original categorical fields from `X`.
- `GridSearchCV` is run on the training data. For the cleanest evaluation, select hyperparameters only within training folds and keep the test set untouched until final evaluation.
- Some cross-validation calls use default model parameters, so their scores are not necessarily cross-validation scores for the tuned model configuration.
- The charts and numeric results are transcribed from the provided artifacts; small differences may occur if library versions or preprocessing differ.
- The current feature matrix may still include columns that are not ideal predictors (for example, check whether any non-feature identifier or target-derived columns remain after preprocessing). Review `X.columns` before presenting.
- For stronger practice, use a `Pipeline` with preprocessing and model fitting inside cross-validation, and report a confusion matrix plus per-class metrics.

## Project files
- `Project 2.py` — original analysis and modelling script.
- `README.md` — this guide.
- `Telco_Churn_Project_Documentation.pdf` — project report, charts, data story, and full source-code appendix.
- The chart images supplied with the project are used in the report.

## Conclusion
The supplied experiment suggests that Logistic Regression, SVM, and the soft-voting ensemble perform similarly on overall accuracy. The ensemble ranks first on accuracy, while Logistic Regression gives the highest reported recall and F1-score among these three. Model choice should therefore reflect the project goal and the relative cost of missing a customer who will churn.
