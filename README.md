# House Price Prediction Using Machine Learning (Myanmar Housing Data)

## Project Overview

This project focuses on **House Price Prediction** using **real housing data collected from Myanmar**, especially from **Yangon housing and condo listings**.

Our group collected real-world housing data from online property platforms in Myanmar. The dataset was carefully **cleaned, processed, and analyzed**, and multiple **machine learning regression models** were trained to predict house prices accurately.

The main aim of this project is to help users **estimate house prices quickly and confidently** using data-driven machine learning techniques.

---

## Group Work & Data Collection

This is a **group research project**.

* Data was collected from **real Myanmar housing sources** (online listings and records).
* Raw data contained missing values, inconsistent formats, and noise.
* We performed **data cleaning, preprocessing, and feature selection** before model training.
* The final dataset represents realistic housing conditions in Myanmar.

---

## Project Objectives

* Develop a machine learning model to predict house prices effectively
* Ensure users can rely on accurate price predictions
* Support quicker and more confident decision-making
* Evaluate model performance using statistical analysis
* Present the research findings clearly through a research poster

---

## Methodology

The project follows a **data analysis and machine learning workflow**:

1. **Data Collection**

   * Real housing data from Myanmar property listings

2. **Data Cleaning & Preprocessing**

   * Handling missing values
   * Encoding categorical features
   * Feature scaling and transformation

3. **Model Training**
   Multiple regression models were trained and compared:

   * Linear Regression
   * Polynomial Regression
   * Ridge Regression
   * Lasso Regression

4. **Model Evaluation**

   * Error comparison
   * Performance analysis
   * Best model selection

5. **Implementation**

   * Final model used for house price prediction
   * Prices predicted in **USD and MMK**

---

## Case Study & Research Approach

* **Case Study**: Yangon housing market
* **Research Type**: Action Research
* **Techniques Used**:

  * Observation
  * Data analysis
  * Questionnaires and records (for understanding housing trends)
  * Personal and public housing records

---

## Project Structure

```
HousePricePrediction/
│
├── Churn.ipynb                         # Model training & analysis notebook
├── Churn_Modelling.csv                # Cleaned housing dataset
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
│
├── research/
│   └── House_Price_Prediction_Research_Poster.pptx
│
└── .gitignore
```

---

## Research Poster Summary

This repository includes a **research poster** created for academic presentation.

### Poster Highlights:

* Real Myanmar housing dataset
* Comparison of multiple regression models
* Model performance analysis
* Key findings and insights
* Practical application for house price estimation

The poster file can be found in the **`research/`** folder.

---

## Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib / Seaborn
* Jupyter Notebook
* Git & GitHub

---

## ▶️ How to Run This Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/KhantZaySoe/house-price-prediction.git
cd house-price-prediction
```

### 2️⃣ Create and Activate Virtual Environment

```bash
python -m venv .venv
```

Activate (Windows):

```bash
.\.venv\Scripts\Activate.ps1
```

### 3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Notebook

```bash
jupyter notebook
```

Open **`Churn.ipynb`** and run cells step by step.

---

## 📈 Results

* The trained model can predict house prices with reasonable accuracy
* Polynomial and regularized regression models performed better on complex patterns
* The results show that machine learning can support housing price estimation in Myanmar

---


## Authors

**Khant Zay Soe**
House Price Prediction (Myanmar Housing Data)

---

## Note

This project is developed for **academic and learning purposes only** and does not represent official property valuations.
