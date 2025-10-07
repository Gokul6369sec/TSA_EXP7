# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 7/10/2025



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

data = pd.read_csv('ai financial.csv')
data = data.groupby('year')['rating'].mean().reset_index()
data['year'] = pd.to_datetime(data['year'], format='%Y')
data = data.set_index('year')

result = adfuller(data['rating'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

x = int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]

lag_order = 13
model = AutoReg(train_data['rating'], lags=lag_order)
model_fit = model.fit()

plt.figure(figsize=(10, 6))
plot_acf(data['rating'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(data['rating'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

start_index = len(train_data) 
end_index = len(data) - 1 

predictions = model_fit.predict(start=start_index, end=end_index, dynamic=False)
predictions.index = test_data.index

mse = mean_squared_error(test_data['rating'], predictions)
print('Mean Squared Error (MSE):', mse)

plt.figure(figsize=(12, 6))
plt.plot(test_data['rating'], label='Test Data - IMDb Rating')
plt.plot(predictions, label='Predictions - IMDb Rating', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Average IMDb Rating')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.grid()
plt.show()
```
### OUTPUT:

GIVEN DATA
<img width="416" height="70" alt="image" src="https://github.com/user-attachments/assets/7618e8f0-1c29-415a-95cb-943f94598b93" />


PACF - ACF
<img width="1326" height="672" alt="image" src="https://github.com/user-attachments/assets/1b00e5b4-8a42-497f-92d6-f679e64c962c" />
<img width="1026" height="596" alt="image" src="https://github.com/user-attachments/assets/c77e5c4f-5550-4787-a2a9-c3786f34a258" />



PREDICTION

FINIAL PREDICTION
<img width="1389" height="721" alt="image" src="https://github.com/user-attachments/assets/770daa76-9612-4c44-b21d-32e742d57bfb" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
