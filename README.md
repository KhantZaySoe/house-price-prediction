# 🏠 Yangon Condo Price Prediction

This project is a simple machine learning web application that predicts condo prices in Yangon based on property features such as area, bedrooms, bathrooms, township, and amenities.  
The system returns the predicted price in both USD and MMK.

---

## 🚀 Features
- Predict condo price using a trained machine learning model  
- Web application built with Flask  
- Uses cleaned housing dataset  
- Shows predicted price in USD and MMK  

---

## 🛠 Tools Used
- Python  
- Flask  
- Scikit-learn  
- Pandas  
- NumPy  
- HTML / CSS  

---

## 📂 Project Structure
- `app.py` – Flask web application  
- `condo_price_model.pkl` – Trained ML model  
- `model_columns.pkl` – Feature columns used during training  
- `X_train_mean.pkl` – Mean values for prediction  
- `FinalCleanedOk.csv` – Cleaned dataset  
- `mytrainingmodel.ipynb` – Model training notebook  
- `requirements.txt` – Project dependencies  

---

## ▶️ How to Run

1. Clone the repository  
   git clone https://github.com/KhantZaySoe/house-price-prediction

2. Install dependencies  
   pip install -r requirements.txt

3. Run the application  
   python app.py

4. Open your browser and go to  
   http://127.0.0.1:5000

---

## 📈 What I Learned
- Data cleaning and preprocessing  
- Regression models (Polynomial, Ridge, Lasso)  
- Building a Flask web application  
- Deploying a machine learning model  
