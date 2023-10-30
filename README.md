This repository contains the code of our [paper](https://link.springer.com/chapter/10.1007/978-981-99-5166-6_35) entitled "
An Exploratory Comparison of LSTM and BiLSTM in Stock Price Prediction". The data we use for experiment can be downloaded [here](https://github.com/quocviethere/An-Exploratory-Comparison-of-LSTM-and-BiLSTM-in-Stock-Price-Prediction/data/AAPL.csv).

# Abstract
Forecasting stock prices is a challenging topic that has been the subject of many studies in the field of finance. Using machine learning techniques, such as deep learning, to model and predict future stock prices is a potential approach. Long Short-Term Memory (LSTM) and Bidirectional Long Short-Term Memory (BiLSTM) are two common deep learning models. The finding of this work is to discover which activation function and which optimization method will influence the performance of the models the most. Also, we implement the comparison of closely related models: vanilla RNN, LSTM, and BiLSTM to discover the best model for stock price prediction. Experimental results indicated that BiLSTM with ReLU and Adam method achieved the best performance in the prediction of stock price.

# Implementation

The source code to implement different models for comparison is given in the `src` folder. To reproduce our result, first you need to clone the repository:

```
git clone https://github.com/quocviethere/An-Exploratory-Comparison-of-LSTM-and-BiLSTM-in-Stock-Price-Prediction
```

Then simply run whichever model provided:

```
python src/bilstm-af.py
```

# Results

Using Adam Optimization and ReLU activation function, we yielded the results as follows:

| Model        | RMSE   | MPE    | MAPE   | MSE    | MAE    | $R^2$ score |
|--------------|--------|--------|--------|--------|--------|----------|
| BiLSTM       | 3.4927 | 0.0028 | 0.0182 | 12.1992 | 2.7009 | 0.8969   |
| LSTM         | 3.6231 | 0.0021 | 0.0198 | 13.1273 | 2.9242 | 0.8891   |
| Vanilla RNN  | 3.6539 | 0.0062 | 0.0197 | 13.3514 | 2.9062 | 0.8872   |


We further investigate the performance of BiLSTM with different activation functions:

|  Activation functions   | RMSE   | MPE    | MAPE   | MSE    | MAE    | $R^2$ score |
|---------|--------|--------|--------|--------|--------|----------|
| ReLU    | 3.4927 | 0.0028 | 0.0182 | 12.1992 | 2.7009 | 0.8969   |
| Tanh    | 3.5385 | 0.0038 | 0.0185 | 12.5210 | 2.7445 | 0.8942   |
| Sigmoid | 3.6059 | -0.0005 | 0.0190 | 13.0031 | 2.8193 | 0.8901   |

and optimization methods:

| Optimization methods | RMSE   | MPE     | MAPE   | MSE    | MAE    | $R^2$ score |
|----------|--------|--------|--------|--------|--------|----------|
| Adam     | 3.4927 | 0.0028 | 0.0182 | 12.1992 | 2.7009 | 0.8969   |
| RMSprop  | 3.7757 | 0.0108 | 0.0205 | 14.2565 | 3.0363 | 0.8795   |
| SGD      | 8.6535 | 0.0255 | 0.0450 | 74.8843 | 6.8549 | 0.3675   |


# Acknowledgement:
This research is funded by University of Economics Ho Chi Minh City (UEH), Vietnam.
