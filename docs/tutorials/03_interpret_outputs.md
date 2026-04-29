# Tutorial 3: Interpret Model Outputs

Each model writes several output files:

- `*.classification_report.csv`: precision, recall, F1, and support.
- `*.confusion_matrix.csv`: class-by-class confusion matrix.
- `*.per_class.csv`: compact per-class metrics.
- `*.threshold_sweep.csv`: coverage and accepted-sample metrics as confidence threshold changes.
- `*.history.csv`: training loss and validation macro-F1 for neural models.
- `*.pth`: trained PyTorch weights for neural models.

For operational use, inspect `*.threshold_sweep.csv`. Higher confidence thresholds reduce coverage but usually increase accepted-sample accuracy. This supports uncertainty-aware mineral identification instead of forcing every spectrum into a class.
