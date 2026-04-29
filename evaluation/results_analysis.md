# Results Analysis

## Confusion Matrix Explanation
- True Negatives (TN): 24052 normal traces correctly predicted as normal.
- False Positives (FP): 35 normal traces incorrectly flagged as anomalies.
- False Negatives (FN): 440 anomaly traces missed by the model.
- True Positives (TP): 308 anomaly traces correctly detected.

## Metric Interpretation
- Precision: 0.8980. This shows how trustworthy anomaly alerts are.
- Recall: 0.4118. This shows how many true anomalies are captured.
- F1 Score: 0.5646. This balances precision and recall.
- ROC-AUC: 0.7401. This reflects overall separability across thresholds.
- Specificity: 0.9985. This indicates normal-event recognition quality.
- False Alarm Rate: 0.0015. Lower values reduce unnecessary alerts.
- Miss Rate: 0.5882. Lower values mean fewer missed anomalies.

## Why Threshold Tuning Helps
The default threshold 0.5 is not always optimal for imbalanced anomaly datasets. Using a tuned threshold (0.6526) allows the classifier to prioritize the desired balance between precision and recall.
In this project, threshold optimization supports better anomaly detection behavior for real-world logs where failure patterns are less frequent than normal patterns.