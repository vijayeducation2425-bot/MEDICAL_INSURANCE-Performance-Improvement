# MEDICAL INSURANCE - Performance Improvement using ANN

## 📋 Project Overview
This project predicts **Medical Insurance Charges** using an **Artificial Neural Network (ANN)**.
The main focus is **Performance Improvement** - the first ANN model gave a very high loss of **258,101,408** because data was not scaled. After applying Scaling, Encoding and Early Stopping, the model performance improved drastically.

## 📊 Dataset
**File:** `medical-charges.csv`
**Rows:** 1338 | **Columns:** 7

| Feature | Type | Description |
| :--- | :--- | :--- |
| age | Numerical | Age of person |
| sex | Categorical | Male / Female |
| bmi | Numerical | Body Mass Index |
| children | Numerical | No. of children |
| smoker | Categorical | Yes / No - Most important feature |
| region | Categorical | Northeast, Northwest, Southeast, Southwest |
| charges | **Target** | Medical charges to predict |

## 🧠 ANN Setup - Architecture

**Model Type:** Sequential ANN - Regression Model

| Layer | Neurons | Activation | Purpose |
| :--- | :--- | :--- | :--- |
| Input Layer | 8 (after encoding) | - | Takes scaled features |
| Hidden Layer 1 | 64 | ReLU | Learns complex patterns |
| Hidden Layer 2 | 32 | ReLU | Learns deeper patterns |
| Hidden Layer 3 | 16 | ReLU | Final feature extraction |
| Output Layer | 1 | Linear | Predicts `charges` (continuous value) |

**Optimizer:** Adam (lr=0.001)
**Loss Function:** Mean Squared Error (MSE)
**Metrics:** MAE, RMSE
**Batch Size:** 32
**Epochs:** 50 (with EarlyStopping)
**Validation Split:** 20%

## ⚙️ Performance Improvement Techniques Used in ANN
1.  **StandardScaler:** Applied on `X` and `y`. This fixed the 25 Cr loss problem.
2.  **One-Hot Encoding:** For `sex`, `smoker`, `region`
3.  **EarlyStopping(patience=5):** Stops training when `val_loss` stops improving, prevents Overfitting
4.  **Dropout (0.2):** Added to avoid overfitting
5.  **Adam Optimizer:** Faster convergence than SGD

## 🛠️ Tech Stack
Python | Jupyter Notebook (Anaconda) | Pandas, NumPy | Matplotlib | Scikit-Learn | TensorFlow / Keras

## 📈 Results
- **Before Improvement:** Loss = 258,101,408, No Scaling
- **After Improvement:** Loss reduced to < 0.5, Val Loss stable
- **Loss Curve:** Plotted using `history.history['loss']` and `history.history['val_loss']`
- **Evaluation:** R2 Score, RMSE, MAE

## ▶️ How to Run
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
jupyter notebook MEDICAL_INSURANCE-Performance Improvement.ipynb
