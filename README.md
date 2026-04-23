# 🧠 Breast Cancer Prediction System using Machine Learning

This project is a **Machine Learning-based Web Application** that predicts whether a breast tumor is **Benign or Malignant** using clinical features. It demonstrates the complete workflow of a real-world ML system — from data preprocessing and model training to deployment using a Flask web interface.

---

## 🚀 Features

- 🔍 Predicts breast cancer (Benign / Malignant)
- 🤖 Uses Logistic Regression for classification
- ⚙️ Feature Scaling using StandardScaler
- 🎯 Feature Selection using SelectKBest (Top 5 Features)
- 📊 Displays Prediction Probability (Confidence Score)
- 🌐 User-friendly web interface using Flask

---

## 🏗️ Project Structure

Breast Cancer Prediction System using Machine Learning Techniques/
│
├── data.csv
├── train_model.ipynb
│
├── model/
│ ├── model.pkl
│ ├── scaler.pkl
│ ├── selector.pkl
│
├── templates/
│ ├── index.html
│ ├── result.html
│
└── app.py


---

## ⚙️ Technologies Used

- Python 🐍  
- NumPy  
- Pandas  
- Scikit-learn  
- Flask  
- HTML / CSS  

---

## 🧪 Machine Learning Workflow

1. **Data Collection**
   - Breast Cancer Wisconsin Dataset

2. **Data Preprocessing**
   - Handling missing values
   - Encoding target variable (B → 0, M → 1)
   - Feature scaling using StandardScaler

3. **Feature Selection**
   - SelectKBest to select top 5 important features

4. **Model Training**
   - Logistic Regression model

5. **Model Evaluation**
   - Accuracy Score (~95%+)
   - Confusion Matrix

6. **Model Deployment**
   - Flask web application

---

## 📊 Model Details

- Algorithm: Logistic Regression  
- Input Features: Top 5 Selected Features  
- Output:
  - 0 → Benign (Non-cancerous)  
  - 1 → Malignant (Cancerous)  

---

