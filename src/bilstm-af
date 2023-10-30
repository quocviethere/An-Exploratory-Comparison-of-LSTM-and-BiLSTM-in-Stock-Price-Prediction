import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf


from sklearn.model_selection import train_test_split

RANDOM_SEED = 1
tf.random.set_seed(RANDOM_SEED)


DATASET_PATH = "/Users/quocviet/Downloads/An-Exploratory-Comparison-of-LSTM-and-BiLSTM-in-Stock-Price-Prediction/src/AAPL.csv"
df = pd.read_csv(DATASET_PATH)
data = df.filter(['Adj Close'])
dataset = data.values

training_data_len = int(np.ceil( len(dataset) * .95 ))

# Scale the data
from sklearn.preprocessing import StandardScaler

# create scaler
scaler = StandardScaler()
# fit and transform in one step
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training_data_len)]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []
window_size = 60

for i in range(window_size, len(train_data)):
    x_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])
    '''
    if i<= window_size + 1:
        print('x_train',x_train)
        print('\ny_train',y_train)
        print()
    '''
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Define the list of activation functions to test
activation_functions = ['sigmoid', 'tanh', 'relu']

# Define a function to build the model with a specific activation function
def build_model(activation):
    input = tf.keras.layers.Input(shape=(x_train.shape[1], 1), name="input")
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, kernel_initializer=tf.initializers.GlorotUniform(seed=RANDOM_SEED))
    )(input)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=False, kernel_initializer=tf.initializers.GlorotUniform(seed=RANDOM_SEED))
    )(x)

    x = tf.keras.layers.Dense(64, activation=activation, name="dense_1")(x)

    output = tf.keras.layers.Dense(1, name="last_dense")(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model

# Loop through each activation function and train the model
for activation_function in activation_functions:
    print(f"Training model with activation function: {activation_function}")
    
    # Rebuild the model with the current activation function
    model = build_model(activation_function)

    epochs = 50
    batch_size = 64

    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    
    # You can save the model, evaluate it, or do any other desired actions here.
    # For example, you can save the model with a different name for each activation function:
    model.save(f'model_{activation_function}.h5')
