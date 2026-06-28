## Calibration Report
- n_total: 10
- MAE (dB, endorsement subset): 3.60  (threshold 8.0, n=5)
- Macro-F1: 1.000  (threshold 0.8)
- Passes gate: True

### Per-kind F1
| Kind | F1 |
|------|-----|
| endorsement | 1.000 |
| flag | 1.000 |
| irrelevant | 1.000 |

### Confusion Matrix (rows = gold, cols = predicted)
| gold \ pred | endorsement | flag | irrelevant |
|---|---|---|---|
| endorsement | 5 | 0 | 0 |
| flag | 0 | 3 | 0 |
| irrelevant | 0 | 0 | 2 |