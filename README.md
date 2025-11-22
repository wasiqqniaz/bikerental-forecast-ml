# Bike Rental Demand Modeling

Machine Learning project analyzing and predicting bike rental demand using **Linear Regression**, **Polynomial Regression**, and **Logistic Regression**.

---

## Project Overview

The goal of this project was to understand the factors that influence bike rental demand and build predictive models using three different ML approaches. The dataset includes weather details, seasonality, and day-level rental information.

To meet the project requirement, I:

* Cleaned and preprocessed the dataset
* Performed Exploratory Data Analysis (EDA)
* Engineered features
* Evaluated multiple machine learning models
* Compared how each model performs on the same dataset

---

## Problem Statement

Bike rental companies rely on accurate demand forecasting to manage inventory, operations, and resource planning.

The challenge was to:

* Analyze what factors impact bike rentals
* Predict **continuous demand** using regression models
* Convert demand into **binary high/low classes** for logistic classification
* Compare performances of Linear, Polynomial, and Logistic Regression
* Understand how feature selection and data preprocessing affect model accuracy

---

## Dataset

**Source:** day.csv (Bike-sharing dataset)
**Main Features:**

* Season, year, month, holiday, weekday
* Temperature, humidity, windspeed
* Total rental count (target for regression)

---

## Project Workflow

This project is organized into three separate scripts for each model:

1. **Linear Regression**
2. **Polynomial Regression**
3. **Logistic Regression**

Each script covers:

* Data cleaning
* Preprocessing
* EDA (numerical + categorical distributions)
* Correlation analysis
* Feature engineering
* Model building
* Model evaluation

---

# 1. Linear Regression (Detailed Work)

### Steps Performed

* Removed unnecessary columns (`instant`, `dteday`, `humidity`, `windspeed`, `atemp`, etc.)
* Avoided **data leakage** by dropping `casual` and `registered` (because they sum up to the target `count`)
* Conducted **extensive EDA**:

  * Histograms
  * Countplots
  * Scatterplots vs. target
  * Heatmap correlation
* Applied one-hot encoding on categorical features
* Scaled features using StandardScaler
* Trained Linear Regression
* Evaluated performance with R² scores

### Feature Selection Experiments

I tested multiple feature combinations to observe performance changes:

* Dropping `registered` & `casual` → **Correct step for leakage**
* Removing `humidity` & `windspeed` → Better performance
* Keeping/Removing `month` & `weekday` (one-hot encoded)

  * Keeping them → Increased complexity & slightly reduced generalization
  * Removing them → Simpler model & improved R²

### Final Linear Regression Results

* **Train R²:** ~0.90+
* **Test R²:** ~0.84–0.88 (depending on final feature selection)

---

# 2. Polynomial Regression (Short Summary)

### Steps

* Loaded cleaned dataset
* Scaled features
* Applied **PolynomialFeatures(degree=2)**
* Trained Linear Regression on polynomial-transformed features

### Performance

Polynomial Regression improved flexibility and captured non-linear relationships.

* **Polynomial R² (Degree=2):** ~0.95+ depending on features used

---

# 3. Logistic Regression (Short Summary)

### Goal

Convert continuous demand into **binary high vs low demand** using the median split.

### Steps

* Created `demand` = 1 (high) or 0 (low) based on median
* Cleaned and selected relevant features
* Applied StandardScaler
* Trained Logistic Regression
* Evaluated using:

  * Accuracy
  * Confusion Matrix
  * Classification Report

### Results

* **Train Accuracy:** ~0.80+
* **Test Accuracy:** ~0.75–0.80
* Good classification performance based on weather & seasonal patterns

---

## Visualizations Included

The project includes multiple visualizations for understanding data:

* Numerical distributions
* Categorical distributions
* Scatterplots with rental count
* Bar charts (average count per category)
* Correlation heatmap

---

## Models Overview

| Model Type                | Target     | Key Steps         | Performance |
| ------------------------- | ---------- | ----------------- | ----------- |
| **Linear Regression**     | Continuous | Scaling + One-hot | R² ~0.85+   |
| **Polynomial Regression** | Continuous | Degree=2 features | R² ~0.95    |
| **Logistic Regression**   | Binary     | Scaling           | Acc ~0.78   |

---

## Tech Stack

* **Python**
* **Pandas, NumPy**
* **Matplotlib, Seaborn**
* **Scikit-learn**

---

## Conclusion

This project successfully builds and compares three machine learning models on the Bike Rental dataset. The analysis shows:

* Linear Regression works well but is limited by linearity
* Polynomial Regression captures more complex relationships
* Logistic Regression is effective for high/low demand classification

This end-to-end pipeline demonstrates solid data preprocessing, feature engineering, visualization, and model evaluation skills.
