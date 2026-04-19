# Log Failure Prediction Using LSTM (Submission Project)

This folder contains the final, modular implementation of a log anomaly detection system based on
EventId sequences and an LSTM classifier.

The pipeline supports:
- training from HDFS trace data,
- threshold-tuned binary anomaly prediction,
- root-cause event identification,
- confidence-based alert levels,
- normal vs anomaly transition visualization,
- online sequence collection from live predictions,
- trigger-based self-learning retraining,
- prediction through CLI, desktop app, Streamlit, and Flask web app.

## 1. Project Objective

Given raw system logs or EventId sequences (E1, E2, E3, ...), predict whether the sequence indicates:
- Normal/Success (0)
- Failure/Anomaly (1)

The model uses sliding windows of length 5 and outputs a probability score. Prediction then uses
a tuned threshold (saved in config) instead of fixed 0.5.

## 2. High-Level Workflow

1. Parse raw logs and map lines to templates.
1. Convert templates to EventIds.
1. Build fixed-length sequence windows.
1. Run LSTM inference.
1. Return:
	- anomaly probability,
	- final class,
	- alert level (NORMAL/WARNING/CRITICAL FAILURE),
	- root cause event + explanation.
1. Optionally visualize normal vs anomaly event transitions.

## 3. Folder Structure

- `data/`: project-local data notes.
- `preprocessing/`: loading, token extraction, and sequence generation logic.
- `model/`: LSTM architecture definition.
- `training/`: end-to-end train/evaluate/orchestrate pipeline.
- `evaluation/`: analysis text and plotting utilities.
- `prediction/`: predictors, template matcher, and interface entry points.
- `webapp/`: Flask UI with HTML/CSS templates.
- `saved_models/`: trained model and encoder artifacts.
- `plots/`: generated evaluation and transition plots.
- `config.json`: runtime configuration (sequence length, threshold, artifact paths).
- `train.py`: training entrypoint.
- `predict.py`: quick CLI prediction entrypoint.

## 4. Key Features (Novel Additions)

### A) Root Cause Event Identification

For anomaly predictions, the system identifies the most suspicious event in the input window by
comparing per-event historical failure association from dataset traces.

Output fields:
- `root_cause_event`
- `root_cause_explanation`

### B) Confidence-Based Alert System

Alert level is derived from anomaly probability:
- Probability < 0.40 -> `NORMAL`
- 0.40 <= Probability <= 0.70 -> `WARNING`
- Probability > 0.70 -> `CRITICAL FAILURE`

Output field:
- `alert_level`

### C) Log Pattern Visualization

The evaluation module builds a directed transition graph from trace data:
- Nodes = EventIds
- Edges = observed transitions
- Normal transitions shown separately from anomaly transitions
- Anomaly transitions highlighted in red

Generated plot:
- `project/plots/event_transition_comparison.png`

The Flask app also renders this graph from:
- `project/webapp/static/event_transition_comparison.png`

## 5. Environment Setup (Windows PowerShell)

From workspace root (one level above `project/`):

```powershell
python -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
```

## 6. Training

Run full training + evaluation + artifact generation:

```powershell
python -m project.train
```

This command will:
- load traces from `dataset/Event_traces.csv`,
- create sequence windows,
- train and evaluate LSTM,
- tune decision threshold,
- save model/encoder/config,
- generate plots (confusion matrix, ROC, learning curves, transition graph),
- generate markdown analysis.

Saved artifacts:
- `project/saved_models/lstm_failure_predictor.keras`
- `project/saved_models/label_encoder.joblib`
- `project/config.json`
- `project/plots/*.png`
- `project/evaluation/results_analysis.md`

## 7. Run Prediction Interfaces

### Quick CLI Prediction

```powershell
python -m project.predict E1 E2 E3 E4 E5
```

Or run without args and enter sequence interactively:

```powershell
python -m project.predict
```

### Manual CLI Predictor (Detailed Output)

```powershell
python -m project.prediction.manual_cli
```

Shows probability, threshold, class, alert level, root-cause details, and unknown EventIds.

### Desktop App (Tkinter)

```powershell
python -m project.prediction.log_input_app
```

Paste raw logs, map to EventIds, and predict from GUI.

### Streamlit Demo

```powershell
streamlit run project/prediction/demo_streamlit.py
```

If `streamlit` command is not recognized:

```powershell
python -m streamlit run project/prediction/demo_streamlit.py
```

### Flask Web Application

```powershell
python -m project.webapp.app
```

Open browser at:
- `http://127.0.0.1:5000`

Web app capabilities:
- paste raw logs,
- template/EventId mapping table,
- prediction summary (probability, class, alert level, root cause),
- root-cause line mapping,
- transition visualization panel.

### Real-Time Monitor (Self-Learning Enabled)

```powershell
python -m project.prediction.realtime_monitor <path_to_live_log_file>
```

This monitor:
- maps each incoming log line to an EventId via `template_matcher.py`,
- builds a sliding window sequence,
- predicts anomaly status,
- appends each sequence to `project/online_dataset.csv`,
- triggers automatic retraining when self-learning rules are met.

## 8. Self-Learning Module

New module:
- `project/self_learning.py`

Online dataset file:
- `project/online_dataset.csv`

Stored format:
- `Event1,Event2,Event3,Event4,Event5,Label`

Labeling logic for online data collection:
- label `1` if `anomaly_probability >= decision_threshold`
- label `0` otherwise

Retraining triggers:
- new sequences >= `retrain_threshold` (default 500)
- or retraining interval reached (default 600 seconds / 10 minutes) with pending new rows

Retraining actions:
- combine original training data and online dataset
- retrain LSTM model on expanded data
- tune threshold again
- overwrite model + encoder artifacts
- update `config.json`

Retraining logs include:
- `New sequences collected: ...`
- `Triggering model retraining...`
- `Model retrained successfully`
- `Updated model saved`

## 9. Input and Output Format

### Expected Event Sequence Input
- Exactly 5 EventIds for direct predictor calls (`sequence_length` from config).
- Example: `E1 E2 E3 E7 E4`

### Typical Prediction Output Fields
- `input_sequence`
- `anomaly_probability`
- `decision_threshold`
- `predicted_failure`
- `alert_level`
- `root_cause_event`
- `root_cause_explanation`
- `unknown_event_ids`

## 10. Common Issues and Fixes

### `config.json not found`
Run training first:

```powershell
python -m project.train
```

### Sequence length error
Provide exactly the configured window length (default 5 EventIds).

### Unknown EventIds in output
Event IDs not seen during training are flagged in `unknown_event_ids`.

### Streamlit command missing
Use `python -m streamlit ...` or install dependencies again.

## 11. Recommended Demo Flow (Viva)

1. Run `python -m project.train` once and show generated plots.
1. Run manual CLI with a sample sequence and explain alert/root cause fields.
1. Run Flask app and demonstrate raw log mapping + prediction + transition graph.

---

This README is aligned with the modular submission code and includes the latest feature extensions.
