# Submission-Ready Project Folder

This folder contains the polished structure for minor project submission and demo.

## Structure
- `data/`
- `preprocessing/`
- `model/`
- `training/`
- `evaluation/`
- `prediction/`
- `saved_models/`
- `plots/`
- `config.json`
- `train.py`
- `predict.py`

## Run Training
```powershell
python -m project.train
```

## Run CLI Prediction
```powershell
python -m project.predict E1 E2 E3 E4 E5
```

## Run Manual CLI Predictor
```powershell
python -m project.prediction.manual_cli
```

## Run Web Application
```powershell
python -m project.webapp.app
```

This web application lets users paste actual logs in the browser, maps them to template EventIds,
and predicts anomaly/failure using the generated EventId sequence.

## Notes
- By default, training reads data from `dataset/Event_traces.csv` in workspace root.
- Training writes model artifacts to `project/saved_models/` and plots to `project/plots/`.
