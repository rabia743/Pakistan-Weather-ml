# Pakistan Weather Analysis & Prediction

A Machine Learning project that analyzes historical weather data from Islamabad, Karachi, and Lahore (2000–2024) and predicts whether football can be played under given weather conditions using the **K-Nearest Neighbors (KNN)** algorithm.

## Project Overview

* Analyze weather trends across Pakistani cities.
* Perform Exploratory Data Analysis (EDA) and visualize rainfall trends.
* Clean data and perform feature engineering.
* Predict `Football_Play` (Yes/No) using KNN.
* Optionally deploy predictions through a Flask web application.

## Dataset

**File:** `after weather_cleaned.csv`

The dataset contains temperature, rainfall, humidity, wind speed, cloud cover, pressure, and other weather features.

**Cities:** Islamabad, Karachi, Lahore
**Time Period:** 2000–2024

##  Technologies Used

* Python, Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn, SciPy
* K-Nearest Neighbors (KNN)
* Flask, Jupyter Notebook
* Git & GitHub

## Machine Learning Model

The KNN classifier uses eight weather features: average temperature, rainfall, wind speed, humidity, cloud cover, minimum temperature, maximum temperature, and dew point.

* **Algorithm:** KNN
* **Neighbors:** 5
* **Distance Metric:** Euclidean
* **Feature Scaling:** StandardScaler
* **Train-Test Split:** 80/20
* **Reported Accuracy:** 93.90%

The model predicts whether football can be played under the given weather conditions.

## Flask Application

An optional Flask web application allows users to enter weather values and receive predictions.

## Author

**Rabia Zafar**

Python Developer | Data Analysis & Machine Learning Enthusiast

---

**Built with Python, Pandas, Scikit-learn, and Flask.**

