# Log Failure Prediction using LSTM - Report Sections

## 1. Dataset Description
The project uses HDFS log trace data from `dataset/Event_traces.csv`. Each row contains:
- `Label`: class label (`Success`/`Normal` or `Fail`/`Anomaly`)
- `Features`: ordered event tokens such as `E5,E22,E11,...`

The objective is binary failure prediction based on event sequence behavior.

## 2. Data Preprocessing
Preprocessing extracts EventIds using regex and normalizes labels into binary targets:
- `Normal/Success` -> 0
- `Fail/Anomaly` -> 1

A label encoder converts EventIds into integer tokens for neural network input.

## 3. Sequence Generation
Each trace is converted into fixed-size sliding windows (length = 5). Every window inherits the parent trace label. This transforms unstructured log traces into sequential supervised samples.

## 4. LSTM Architecture
Model design:
- Embedding layer for token representation
- LSTM layer for temporal pattern learning
- Dense layers with dropout for nonlinear classification
- Sigmoid output for anomaly probability

Loss: Binary cross-entropy, Optimizer: Adam.

## 5. Training Procedure
Training flow includes:
- optional balanced subset selection
- train/validation/test split
- minority class oversampling on train windows
- early stopping and LR reduction callbacks
- threshold tuning from validation probabilities

## 6. Evaluation Metrics
Reported metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC

Generated visuals:
- confusion matrix heatmap
- ROC curve
- train vs validation accuracy
- train vs validation loss

## 7. Results and Discussion
The confusion matrix quantifies true/false positives and negatives. Precision indicates alert reliability, recall indicates anomaly capture rate, and F1 balances both.

Threshold tuning improves detection behavior compared to fixed 0.5 by adapting decision boundary to dataset imbalance and operational objective (e.g., high precision target).

Detailed generated analysis is available in `project/evaluation/results_analysis.md` after training.

## 8. System Architecture
See `project/architecture_diagram.md` for pipeline diagram:
Log Dataset -> Parsing -> Event Extraction -> Sequence Generation -> LSTM -> Evaluation -> Prediction.

## 9. Conclusion
The project delivers an end-to-end sequence-based log anomaly prediction pipeline with saved artifacts, interpretable evaluation outputs, and live inference interfaces.

## 10. Future Work
- Add attention or Transformer variants for long-range dependencies.
- Introduce automated retraining and drift monitoring.
- Extend unknown event handling with semantic nearest-event mapping.
- Provide REST API deployment for production integration.

## Novel Features to Highlight
- Threshold optimization instead of fixed threshold.
- Unknown event detection during prediction.
- Sequence-based log learning using LSTM.
- Manual and Streamlit-based anomaly testing interface.
- Full ML pipeline with reproducible saved artifacts.
