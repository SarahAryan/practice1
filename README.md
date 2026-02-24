# 🚀 CI/CD Failure Prediction using LSTM

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)


A deep learning project that predicts **CI/CD pipeline failures** based on historical workflow execution data using a multi-layer LSTM model built with PyTorch.

---

## ✨ Features

- 📊 Reads CI/CD execution data from CSV
- 🧹 Automatic data cleaning and preprocessing
- 🔄 Sequential sample generation per repository
- 🔢 Categorical feature encoding using `LabelEncoder`
- ⚖️ Stratified train/test split
- 🧠 Two-layer LSTM neural network (128 hidden units)
- 🎯 Handles class imbalance using weighted loss
- 📈 Evaluation metrics:
  - Accuracy
  - F1-score
  - Confusion Matrix
  - Classification Report


---

## ⚙️ Installation

### 1️⃣ Clone Repository

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy torch scikit-learn matplotlib
```

If using **Google Colab**, most packages are pre-installed.

---

## 📂 Dataset Requirements

Place your dataset at for using colab:

```
/content/drive/MyDrive/sample_records.csv
```

Required columns:

- `repository_name`
- `metadata.workflow_id`
- `metadata.event`
- `metadata.actor.login`
- `metadata.run_started_at`
- `metadata.conclusion`

The `metadata.conclusion` column must contain values such as:

- `failure`
- `success`

---

## ▶️ How to Run

### ✅ Google Colab

1. Upload the script or notebook.
2. Mount Google Drive (handled automatically).
3. Run all cells.

### ✅ Local Execution

1. Remove the Google Drive mounting section.
2. Update file path:

```python
input_file = "path/to/sample_records.csv"
```

3. Run:

```bash
python main.py
```

---

## 🧠 Model Architecture

```
Input Features (Encoded Sequences)
        ↓
2-Layer LSTM (128 Hidden Units, Dropout=0.2)
        ↓
Fully Connected Layer
        ↓
Sigmoid Output (Failure Probability)
```

- Loss Function: `BCEWithLogitsLoss`
- Optimizer: `Adam`
- Epochs: 100
- Class imbalance handled using `pos_weight`

---

## 📊 Output Example

```
==============================
Final Results:

Confusion Matrix:
[[TN FP]
 [FN TP]]

Classification Report:
              precision    recall    f1-score
Success        0.xx        0.xx       0.xx
Failure        0.xx        0.xx       0.xx
==============================
```

---

## 👩‍💻 Credits

This project was designed and develoed by **Sarah aryan**  


---

## 📜 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project with proper attribution.
