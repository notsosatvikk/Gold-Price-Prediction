# **Gold Price Prediction**

## **Overview**  
This project involves predicting gold prices using machine learning models. By analyzing historical data and various influencing factors, the project builds regression models to provide accurate price predictions. The models used include Linear Regression, k-Nearest Neighbors (k-NN), and Random Forest Regression.

---

## **Features**  
- Exploratory Data Analysis (EDA) with visualizations.  
- Feature-target correlation analysis.  
- Outlier handling for robust model training.  
- Implementation of multiple regression models:
  - **Linear Regression**  
  - **k-Nearest Neighbors Regression**  
  - **Random Forest Regression**  
- Model comparison based on R-squared scores.  
- Interactive predictions based on user input.  

---

## **Requirements**  
Install the following dependencies:  
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## **Dataset**  
- The project uses a dataset (`gld_price_data.csv`) containing historical gold prices and related financial features.  
- Ensure the dataset is placed in the project directory.

---

## **File Structure**  
```
Gold Price Prediction/
│
├── gld_price_data.csv       # Dataset
├── gold_price_prediction.py # Main script
├── README.md                # Project documentation
```

---

## **How to Run**  

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/your-username/gold-price-prediction.git
   cd gold-price-prediction
   ```

2. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Script**:  
   ```bash
   python gold_price_prediction.py
   ```

4. **Make Predictions**:  
   - Enter feature values when prompted to get the predicted gold price.

---

## **Exploratory Data Analysis (EDA)**  
The project includes detailed visualizations to understand the dataset:  
- **Scatter Plot**: Gold price vs. SPX.  
- **Histogram**: Distribution of gold prices.  
- **Bar Plot**: SLV vs. Gold prices.  
- **Box Plot**: Identify outliers in gold prices.  
- **Heatmap**: Correlation between features and gold price.  

---

## **Models Implemented**  

### 1. **Linear Regression**  
A basic regression model to understand linear relationships between features and gold price.  
- **R-squared Error**: `0.XXX` (example value).  

### 2. **k-Nearest Neighbors (k-NN) Regression**  
A non-parametric method for regression using proximity-based predictions.  
- **R-squared Error**: `0.XXX` (example value).  

### 3. **Random Forest Regression**  
An ensemble learning method that combines multiple decision trees for robust predictions.  
- **R-squared Error**: `0.XXX` (example value).  

---

## **Model Evaluation**  
The models are evaluated and compared using R-squared scores. The Random Forest Regressor outperforms other models and is selected as the best model.  
- **Visualization**: Bar plot comparing R-squared errors for all models.  

---

## **Future Enhancements**  
- Implement additional machine learning models for comparison.  
- Use hyperparameter tuning for optimal model performance.  
- Incorporate live financial data for real-time predictions.  

---
