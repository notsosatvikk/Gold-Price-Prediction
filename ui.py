import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
import tkinter as tk
from tkinter import ttk

# Loading the csv data to a Pandas DataFrame
gold_data = pd.read_csv('Project/gld_price_data.csv')

# Splitting the Features and Target
X = gold_data.drop(['Date','GLD'], axis=1)
Y = gold_data['GLD']

# Splitting into Training data and Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model Training: Random Forest Regressor
random_forest_regressor = RandomForestRegressor(n_estimators=100)
random_forest_regressor.fit(X_train, Y_train)

# Create UI
def predict_gold_price():
    user_input = []
    for feature in X.columns:
        value = float(entry_vars[feature].get())
        user_input.append(value)

    user_input = np.array(user_input).reshape(1, -1)
    user_prediction = random_forest_regressor.predict(user_input)
    
    # Create a popup window
    popup = tk.Toplevel()
    popup.title("Prediction Result")
    
    # Create a custom style for the background color
    style = ttk.Style(popup)
    style.configure('Custom.TLabel', background='black', foreground='gold')
    
    result_label = ttk.Label(popup, text="Predicted value: {:.2f}".format(user_prediction[0]), style='Custom.TLabel')
    result_label.pack(padx=20, pady=10)
    
    close_button = ttk.Button(popup, text="Close", command=popup.destroy)
    close_button.pack(pady=5)

root = tk.Tk()
root.title("Gold Price Predictor")

# Create a custom style for the background color
style = ttk.Style()
style.configure('Custom.TFrame', background='black')

frame = ttk.Frame(root, padding="20", style='Custom.TFrame')
frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

entry_vars = {}
for i, feature in enumerate(X.columns):
    ttk.Label(frame, text=feature+":", foreground="gold", background="black").grid(column=0, row=i, sticky=tk.W)
    entry_vars[feature] = tk.StringVar()
    ttk.Entry(frame, textvariable=entry_vars[feature]).grid(column=1, row=i, padx=10, pady=5)

predict_button = ttk.Button(frame, text="Predict", command=predict_gold_price)
predict_button.grid(column=0, row=len(X.columns), columnspan=2, pady=10)

root.mainloop()
