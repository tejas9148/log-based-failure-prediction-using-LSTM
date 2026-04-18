# System Architecture Diagram

```mermaid
flowchart LR
    A[Log Dataset] --> B[Log Parsing]
    B --> C[Event ID Extraction]
    C --> D[Sliding Window Sequence Generation]
    D --> E[LSTM Model]
    E --> F[Evaluation]
    F --> G[Prediction System]
```

Pipeline summary:
- Log data is parsed to extract EventIds from traces.
- Event sequences are transformed into fixed-length windows.
- LSTM learns sequential failure patterns.
- Evaluation provides confusion matrix, ROC, and key metrics.
- Prediction system supports CLI and Streamlit demo.
