# Effect of Raman Spectral QC Policy on Reviewer-Requested Models

The hard-only policy corresponds to the previous 900-spectrum reference-domain comparison after excluding halides. The balanced policy additionally excludes low-confidence RRUFF identifications and continuum/bandless spectra, but keeps spike-flagged spectra in the main set with QC flags. The strict policy excludes all manual-review flags and is included as an over-filtering sensitivity test.

| model                | qc_policy                      |   accuracy |   macro_f1 |   weighted_f1 |
|:---------------------|:-------------------------------|-----------:|-----------:|--------------:|
| standard_transformer | hard_only_halides_excluded_900 |      0.759 |      0.674 |         0.749 |
| standard_transformer | balanced_qc_794                |      0.774 |      0.665 |         0.776 |
| standard_transformer | strict_qc_637                  |      0.747 |      0.651 |         0.745 |
| mst                  | hard_only_halides_excluded_900 |      0.767 |      0.637 |         0.768 |
| mst                  | balanced_qc_794                |      0.765 |      0.636 |         0.763 |
| mst                  | strict_qc_637                  |      0.692 |      0.612 |         0.711 |
| pca_svm              | hard_only_halides_excluded_900 |      0.722 |      0.609 |         0.713 |
| pca_svm              | balanced_qc_794                |      0.765 |      0.632 |         0.762 |
| pca_svm              | strict_qc_637                  |      0.725 |      0.669 |         0.709 |
| cnn                  | hard_only_halides_excluded_900 |      0.662 |      0.553 |         0.646 |
| cnn                  | balanced_qc_794                |      0.722 |      0.65  |         0.709 |
| cnn                  | strict_qc_637                  |      0.538 |      0.469 |         0.513 |
| pls_da               | hard_only_halides_excluded_900 |      0.549 |      0.376 |         0.509 |
| pls_da               | balanced_qc_794                |      0.591 |      0.476 |         0.583 |
| pls_da               | strict_qc_637                  |      0.538 |      0.416 |         0.52  |

Interpretation: balanced QC improves the CNN baseline substantially relative to the earlier hard-only table and keeps the MST close to the original performance, while strict QC removes too many spectra and reduces neural-network performance. Therefore, balanced QC is the most defensible main-data policy, with strict QC reported only as sensitivity analysis.
