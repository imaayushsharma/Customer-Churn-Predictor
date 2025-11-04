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
2. Create and activate a virtual environment
bash
Copy code
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
If you don’t have a requirements.txt, you can install them manually:

bash
Copy code
pip install streamlit tensorflow scikit-learn pandas numpy
▶️ Usage
Run the Streamlit app:
bash
Copy code
streamlit run streamlit_churn_frontend.py
Then open in your browser:
arduino
Copy code
http://localhost:8501
🧾 Input Features
Feature	Description	Example
CreditScore	Customer’s credit score	715
Geography	Country (France, Germany, Spain)	France
Gender	Male or Female	Male
Age	Age of the customer	37
Tenure	Years of association with the bank	5
Balance	Account balance	84532.45
NumOfProducts	Number of bank products used	2
HasCrCard	Has a credit card (1 = Yes, 0 = No)	1
IsActiveMember	Active customer (1 = Yes, 0 = No)	0
EstimatedSalary	Customer’s annual salary	112450.30

📈 Output
The app will display:

Prediction Result: Exited or Not Exited

Prediction Probability: Model’s confidence score

Progress bar visualizing churn likelihood

🧰 Tech Stack
Frontend: Streamlit

Backend: TensorFlow / Keras

Data Processing: scikit-learn, pandas, numpy

Model Type: Deep Neural Network (Binary Classification)

📦

🧑‍💻 Author
Ayush Sharma
Associate Data Analyst @ GlobalLogic
📧 official.aayushsharmaa@gmail.com

🔗 LinkedIn
 | GitHub
