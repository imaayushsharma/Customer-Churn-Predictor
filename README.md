# 🧠 Customer Churn Prediction Web App

A professional, interactive **Streamlit web application** that predicts whether a bank customer is likely to **exit (churn)** or **stay**, based on key customer attributes.  
The project uses a pre-trained **Deep Learning model (Keras)** and several preprocessing components such as **LabelEncoder**, **OneHotEncoder**, and **StandardScaler**.

---

## 🚀 Project Overview

This project allows users to input customer information and instantly receive a churn prediction with the model’s confidence score.  
It combines **machine learning**, **data preprocessing**, and a clean **frontend interface built with Streamlit**.

---

## 🧩 Features

- 🎨 **Professional Streamlit UI** – Intuitive, responsive, and well-structured layout  
- 📦 **Pretrained Model Integration** – Loads Keras model (`model.h5`) and preprocessing artifacts (`.pkl` files)  
- ⚙️ **Automatic Preprocessing** – Handles categorical encoding, scaling, and one-hot transformation internally  
- 📊 **Prediction Probability** – Displays churn probability and visual progress bar  
- 🧾 **Input Summary Panel** – Displays all user-entered data for quick review  
- 🔒 **Error Handling** – Graceful handling if model or encoders are missing  

---

## 🧠 Model Artifacts

The following files are required for the app to run successfully:

| File | Description |
|------|--------------|
| `model.h5` | Trained TensorFlow/Keras model |
| `label_encoder_gender.pkl` | Encodes “Gender” column |
| `ohe_encoder_geography.pkl` | One-hot encoder for “Geography” column |
| `scaler.pkl` | StandardScaler for numerical features |
| `streamlit_churn_frontend.py` | Streamlit frontend app |

Place all of these files in the same project directory.

---

## ⚙️ Installation

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/churn-prediction-app.git
cd churn-prediction-app
