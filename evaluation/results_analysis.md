# Results Analysis

## Confusion Matrix Explanation
- True Negatives (TN): 13547 normal traces correctly predicted as normal.
- False Positives (FP): 1990 normal traces incorrectly flagged as anomalies.
- False Negatives (FN): 7535 anomaly traces missed by the model.
- True Positives (TP): 6219 anomaly traces correctly detected.

## Metric Interpretation
- Precision: 0.7576. This shows how trustworthy anomaly alerts are.
- Recall: 0.4522. This shows how many true anomalies are captured.
- F1 Score: 0.5663. This balances precision and recall.
- ROC-AUC: 0.7039. This reflects overall separability across thresholds.
- Specificity: 0.8719. This indicates normal-event recognition quality.
- False Alarm Rate: 0.1281. Lower values reduce unnecessary alerts.
- Miss Rate: 0.5478. Lower values mean fewer missed anomalies.

## Why Threshold Tuning Helps
The default threshold 0.5 is not always optimal for imbalanced anomaly datasets. Using a tuned threshold (0.4455) allows the classifier to prioritize the desired balance between precision and recall.
In this project, threshold optimization supports better anomaly detection behavior for real-world logs where failure patterns are less frequent than normal patterns.