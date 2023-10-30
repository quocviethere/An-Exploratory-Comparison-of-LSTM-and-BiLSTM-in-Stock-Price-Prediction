import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RANDOM_SEED = 1
tf.random.set_seed(RANDOM_SEED)


DATASET_PATH = '/Users/quocviet/Downloads/An-Exploratory-Comparison-of-LSTM-and-BiLSTM-in-Stock-Price-Prediction/data/AAPL.csv'
df = pd.read_csv(DATASET_PATH)

data = df.filter(['Adj Close'])

dataset = data.values
training_data_len = int(np.ceil( len(dataset) * .95 ))


# create scaler
scaler = StandardScaler()
# fit and transform in one step
scaled_data = scaler.fit_transform(dataset)

# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len)]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []
window_size = 60

for i in range(window_size, len(train_data)):
    x_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])
    if i<= window_size + 1:
        print('x_train',x_train)
        print('\ny_train',y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


def build_model():
    input = tf.keras.layers.Input(shape=(x_train.shape[1], 1), name ="input")
    x = tf.keras.layers.SimpleRNN(128, 
                                  return_sequences=True, 
                                  kernel_initializer=tf.initializers.GlorotUniform(seed=RANDOM_SEED))(input)
    x = tf.keras.layers.SimpleRNN(64, 
                                  return_sequences=False, 
                                  kernel_initializer=tf.initializers.GlorotUniform(seed=RANDOM_SEED))(x)
    x = tf.keras.layers.Dense(32, activation="relu", name ="dense_1")(x)
    output = tf.keras.layers.Dense(1, name="last_dense")(x)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model

model = build_model()

epochs = 50
batch_size = 64

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, batch_size=batch_size ,epochs=epochs)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - window_size: ]
print ('len(test_data):', len(test_data))

# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(window_size, len(test_data)):
    x_test.append(test_data[i-window_size:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

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

print(f'RMSE: {rmse(y_test, predictions)}')
print(f'MPE: {mpe(y_test, predictions)}')
print(f'MAPE: {mape(y_test, predictions)}')
print(f'MSE: {mse(y_test, predictions)}')
print(f'MAE: {mae(y_test, predictions)}')
from sklearn.metrics import r2_score
print(f'R2 Score: {r2_score(y_test, predictions)}')