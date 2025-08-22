# DLI_Assignment
DLI Group Assignment

## 7. Results

### Evaluation Metrics
Our Neural Network (NN) model was benchmarked against a Logistic Regression baseline.  
Below is a summary of key evaluation metrics:

| Model                  | F1 Score | ROC-AUC | Accuracy |
|-------------------------|----------|---------|----------|
| Logistic Regression     | 0.87     | 0.90    | 0.88     |
| Neural Network (ours)   | 0.93     | 0.96    | 0.94     |

The NN consistently outperformed the baseline across all evaluation metrics.

### Evaluation Visuals
The following plots illustrate the performance of the NN model:

- **Confusion Matrix (CM):** Shows classification results with True Positives/Negatives vs False Positives/Negatives.
- **ROC Curve:** Demonstrates model separability with AUC ≈ 0.96.
- **Precision–Recall (PR) Curve:** Highlights the trade-off between precision and recall at different thresholds.
- **Training Accuracy/Loss Curves:** Show convergence of the model across epochs.

<p align="center">
  <img src="figures/confusion_matrix.png" width="400"/>  
  <img src="figures/roc_curve.png" width="400"/>  
  <img src="figures/pr_curve.png" width="400"/>  
  <img src="figures/accuracy_loss.png" width="400"/>  
</p>
