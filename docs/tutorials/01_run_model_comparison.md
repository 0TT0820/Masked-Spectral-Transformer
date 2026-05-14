# Tutorial 1: Run a Model Comparison

From the repository root:

```bash
python src/train_model_comparison.py --models pca_svm pls_da random_forest --baseline poly --chemometric-stride 8 --no-augment
```

The script creates a new run directory under:

```text
results/reproduction_runs/
```

Open `model_comparison_summary.csv` to inspect model-level accuracy, macro-F1, and weighted-F1.

For the full deep-learning comparison:

```bash
python src/train_model_comparison.py --models cnn standard_transformer mst --epochs 180 --batch-size 16 --lr 1e-4 --baseline poly --no-augment
```
