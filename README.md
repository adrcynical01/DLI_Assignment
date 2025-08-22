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

## 8. Making Predictions (Interactive CLI)

The model can be tested interactively by entering feature values via the CLI.

- Each of the **30 input features** must be entered as `-1`, `0`, or `1`.  
  (e.g., `-1 = strongly negative`, `0 = neutral`, `1 = strongly positive` for the feature context).

- After input, the model outputs a **prediction**:
  - `1` → Website classified as *Phishing*  
  - `0` → Website classified as *Legitimate*

### Example Run
```bash
$ python predict_cli.py

Enter feature 1 (-1/0/1): 1
Enter feature 2 (-1/0/1): 0
...
Enter feature 30 (-1/0/1): -1

[INFO] Model prediction: PHISHING SITE (1)



---

### **Commit 3: Add Usage Instructions for Saved Model + Streamlit App**

```markdown
## 9. Deployment & Usage

### 9.1 Loading the Saved Model
The trained model is saved as `phishing_model.keras`.  
You can reload it in Python using TensorFlow:

```python
from tensorflow.keras.models import load_model

# Load saved model
model = load_model("phishing_model.keras")

# Make predictions
pred = model.predict(X_test)

9.2 Streamlit Web App

We provide a lightweight Streamlit application for user interaction.

Running the App

Install dependencies:

pip install -r requirements.txt


(Make sure streamlit, tensorflow, numpy, and scikit-learn are installed.)

Run the app:

streamlit run app.py


Open the local URL provided by Streamlit (default: http://localhost:8501).

9.3 Example App Features

Upload CSV with website features → get phishing/legit prediction.

Manual entry of 30 features via sliders.

Displays evaluation figures (CM, ROC, PR) for transparency.
