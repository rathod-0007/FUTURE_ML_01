# FUTURE_ML_01


# 🛒 Sales Forecasting for Retail Business

This project focuses on building a machine learning pipeline for **sales forecasting** using time series analysis and regression techniques. The aim is to predict future sales trends for a retail business and evaluate the model's accuracy using standard performance metrics.

---

## 📌 Project Objectives

- Predict daily/weekly/monthly sales using historical data
- Explore both time series (Prophet) and feature-based regression (Random Forest, Scikit-learn)
- Perform hyperparameter tuning to improve accuracy
- Visualize actual vs predicted sales to assess performance

---

## 🔧 Tools & Technologies

- **Programming Language:** Python
- **Notebook Environment:** Google Colab
- **Libraries:**  
  - `pandas`, `numpy` – Data handling  
  - `matplotlib`, `seaborn` – Visualization  
  - `prophet` – Time series forecasting  
  - `scikit-learn` – Regression models, evaluation, tuning  

---

## 📁 Dataset

- Local CSV file: `sales_data_sample.csv`  
- Format: Includes columns like `ORDERDATE`, `SALES`, `STATUS`, `CUSTOMER`, `PRODUCT`, etc.  
- Assumption: Dates and numeric sales values cleaned and converted properly

---

## 🚀 How to Run

1. **Clone this repository**  
   ```bash
   git clone https://github.com/rathod-0007/FUTURE_ML_01.git
   cd FUTURE_ML_01
   ```

2. **Upload `sales_data_sample.csv` to Colab environment**

3. **Open `sales_forecasting.ipynb` in Google Colab**

4. **Run the notebook step by step**

---

## ✅ Project Workflow

### Step 1: Load and Inspect Dataset
```python
df = pd.read_csv('sales_data_sample.csv', encoding='latin1')
```

### Step 2: Clean Dates & Sales Columns
- Parse `OrderDate`
- Handle missing/invalid values

### Step 3: Prophet Forecasting
- Use Facebook Prophet for baseline time series model
- Forecast next 30/60/90 days
- Plot trend, seasonality, forecast

### Step 4: Evaluate Prophet Accuracy
```python
mae = mean_absolute_error(actual['y'], predicted['yhat'])
```

### Step 5: Feature Engineering for Regression
- Time index, lag features, rolling averages

### Step 6: Train Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor
model_rf.fit(X_train, y_train)
```

### Step 7: Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, ...)
```

### Step 8: Evaluate Models
- MAE, RMSE, R² Score
- Plot predictions vs actual sales

---

## 📊 Results & Evaluation

| Model              | MAE    | RMSE   | R² Score |
|-------------------|--------|--------|----------|
| Prophet            | ~1423   | ~1834   | ~0.01    |
| Random Forest      | ~1526   | ~1949   | ~0.12    |
| Tuned RF (GridSearchCV) | **~1507** | **~1942** | **~0.12** |

---

## 📈 Visual Output Example

![Forecast Plot](https://github.com/rathod-0007/FUTURE_ML_01/blob/5495d27e8fdb2b023b8d9266c150560bc81d9b0a/Sales%20Forecast.png)  
*Forecast trend using Prophet*

![Forecast Plot]([assets/forecast_plot.png](https://github.com/rathod-0007/FUTURE_ML_01/blob/5495d27e8fdb2b023b8d9266c150560bc81d9b0a/Sales_Forecast.png))  
*Forecast trend using Prophet*

![RF Plot]([assets/rf_prediction.png](https://github.com/rathod-0007/FUTURE_ML_01/blob/5495d27e8fdb2b023b8d9266c150560bc81d9b0a/RF%20Actual%20vs%20Predicted.png))  
*Random Forest Predictions vs Actual*

![Tuned RF Plot]([assets/tuned_rf_prediction.png](https://github.com/rathod-0007/FUTURE_ML_01/blob/5495d27e8fdb2b023b8d9266c150560bc81d9b0a/Tuned%20RF.png))  
Tuned Random Forest Predictions vs Actual*

---

## 📝 Conclusion

This project demonstrated how to:
- Use Prophet for baseline sales forecasting
- Improve performance using feature-based regression
- Tune models for better prediction accuracy

---

## 📬 Contact

For queries or collaborations:

- Name: Pavan Kumar Naik Rathod
- Email: [rathodpavan2292@gmail.com]  
- LinkedIn: [https://www.linkedin.com/in/rathod-pavan-kumar/](https://www.linkedin.com/in/rathod-pavan-kumar/)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
