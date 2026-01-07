# House Price Prediction System (Yangon)

This project is a machine learning–based web application that predicts house prices in Yangon using property features such as area, bedrooms, bathrooms, amenities, and township.



##  Features
- Predicts house prices using a trained ML model
- Compares multiple regression models (Polynomial, Ridge, Lasso)
- Uses the best-performing model for final prediction
- Web-based interface built with HTML, Tailwind CSS, and JavaScript
- Backend developed using Flask (Python)

---

##  Machine Learning Models
- Polynomial Regression (Best Model)
- Ridge Regression
- Lasso Regression

Model performance is evaluated using MAE, MSE, RMSE, and R² score.

---

##  Tech Stack
- Python
- Flask
- Scikit-learn
- Pandas & NumPy
- HTML
- Tailwind CSS
- JavaScript

---

## Project Files Summary

### app.py
Main Flask application.
Runs the web app, takes user input (area, rooms, township, amenities), loads the trained model, and returns house price predictions.

### condo_price_model.pkl
Final trained machine learning model.
Created from cleaned housing data and used by app.py to predict prices.

### model_columns.pkl
Stores the feature names used during training.
Ensures correct input format during prediction.

### X_train_mean.pkl
Contains mean values of training data.
Used to handle missing or optional user inputs.

### FinalCleanedOk.csv
Cleaned dataset used for model training and evaluation.
Missing values and outliers were handled.

### mytrainingmodel.ipynb
Model development notebook.
Includes data preprocessing, model training (Polynomial, Ridge, Lasso), evaluation, and model selection.

### Requirements

This project uses Python and several machine learning libraries.
All required dependencies are listed in requirements.txt

How to install dependencies
After cloning the repository and activating a virtual environment, run:
pip install -r requirements.txt

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/KhantZaySoe/house-price-prediction.git
cd house-price-prediction
