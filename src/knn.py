import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Read in the data from the CSV file
DATASET_PATH = '/Users/quocviet/Downloads/An-Exploratory-Comparison-of-LSTM-and-BiLSTM-in-Stock-Price-Prediction/data/AAPL.csv'
data = pd.read_csv(DATASET_PATH,index_col=0)

# Split the data into input features (X) and the target variable (y)
X = data.drop('Adj Close', axis=1)
y = data['Adj Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the KNN model and fit it to the training data
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

# Use the model to make predictions on the test data
y_pred = model.predict(X_test)

def mae(y_true, y_pred):
    mae = np.mean(np.abs((y_true - y_pred)))

    return mae

def mse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)

    return mse

def rmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true-y_pred)**2))

    return rmse

def mpe(y_true, y_pred):
    mpe = np.mean((y_true-y_pred) / y_true)

    return mpe
def mape(y_true, y_pred):
    mape = np.mean(np.abs((y_true-y_pred)) / y_true)
    return mape

predictions = model.predict(X_test)

print(f'RMSE: {rmse(y_test, predictions)}')
print(f'MPE: {mpe(y_test, predictions)}')
print(f'MAPE: {mape(y_test, predictions)}')
print(f'MSE: {mse(y_test, predictions)}')
print(f'MAE: {mae(y_test, predictions)}')
from sklearn.metrics import r2_score
print(f'R2 Score: {r2_score(y_test, predictions)}')