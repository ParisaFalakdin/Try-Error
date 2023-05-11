#Try and error code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

our_data = pd.read_csv('data-our.csv')
reference_data = pd.read_csv('data-reference.csv')
our_data.head()
reference_data.head()

# calculate metrics
MAE = np.mean(np.abs(our_data['NH3_cal'] - reference_data['NH3']))
RMSE = np.sqrt(np.mean((our_data['NH3_cal'] - reference_data['NH3'])**2))
r = np.corrcoef(our_data['NH3_cal'], reference_data['NH3'])[0, 1]
R2 = r**2

print('MAE:', MAE)
print('RMSE:', RMSE)
print('r:', r)
print('R^2:', R2)

plt.scatter(x=our_data['NH3_cal'], y=reference_data['NH3'] )
plt.xlabel('Calibrated our NH3 (µg/m3)')
plt.ylabel('Referenced Data NH3 (µg/m3)')
plt.title('Calibrated vs Referenced Data Measurements')
plt.show()

NH3Calc = pd.concat([our_data['datetime'], our_data['NH3'],our_data['NH3_cal'], reference_data['NH3']], axis=1)

#columns renamed
NH3Calc.columns = ['datetime', 'NH3_OP_nA', 'NH3_OP_Conc','NH3_Ref_Conc']

#outliers were removed based on visual inspection
NH3Calc = NH3Calc[(NH3Calc['NH3_OP_Conc'] < 60)]
NH3Calc = NH3Calc[(NH3Calc['NH3_Ref_Conc'] < 170)]

#checking how many data (rows) were removed
NH3Calc.head()

NH3Calc = NH3Calc[(NH3Calc['NH3_OP_nA'] < 30)]
NH3Calc = NH3Calc[(NH3Calc['NH3_Ref_Conc'] < 170)]


plt.scatter(x=NH3Calc['NH3_OP_nA'], y=NH3Calc['NH3_Ref_Conc'] )
plt.xlabel('Calibrated our NH3 (µg/m3)')
plt.ylabel('Referenced Data NH3 (µg/m3)')
plt.title('Calibrated vs Referenced Data Measurements')
plt.show()

NH3Calc.shape[0]

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#polynomial with new dataframe

poly = PolynomialFeatures(degree=3)
X = poly.fit_transform(NH3Calc[['NH3_OP_nA']])
model = LinearRegression().fit(X, NH3Calc['NH3_Ref_Conc'])
y_pred = model.predict(X)

MAE_new = np.mean(np.abs(y_pred - NH3Calc['NH3_Ref_Conc']))
RMSE_new = np.sqrt(np.mean((y_pred - NH3Calc['NH3_Ref_Conc'])**2))
r_new = np.corrcoef(y_pred, NH3Calc['NH3_Ref_Conc'])[0, 1]
R2_new = r_new**2

print('MAE:', MAE_new)
print('RMSE:', RMSE_new)
print('r:', r_new)
print('R^2:', R2_new)

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# SVR
scaler = StandardScaler()
X = scaler.fit_transform(NH3Calc[['NH3_OP_nA']])
y = NH3Calc['NH3_Ref_Conc']
model = SVR(kernel='rbf', C=10, gamma=0.1, epsilon=0.1)
model.fit(X, y)
y_pred = model.predict(X)

MAE_SVR = np.mean(np.abs(y_pred - NH3Calc['NH3_Ref_Conc']))
RMSE_SVR = np.sqrt(np.mean((y_pred - NH3Calc['NH3_Ref_Conc'])**2))
r_SVR = np.corrcoef(y_pred, NH3Calc['NH3_Ref_Conc'])[0, 1]
R2_SVR = r_SVR**2

print('MAE_SVR :', MAE_SVR )
print('RMSE_SVR :', RMSE_SVR )
print('r_SVR:', r_SVR)
print('R^2_SVR:', R2_SVR)

plt.scatter(x=y_pred, y=NH3Calc['NH3_Ref_Conc'] )
plt.show()

#try exponential regression

X = NH3Calc[['NH3_OP_nA']]
y = np.log(NH3Calc['NH3_Ref_Conc'])
model = LinearRegression().fit(X, y)
y_pred = np.exp(model.predict(X))

MAE_ER = np.mean(np.abs(y_pred - NH3Calc['NH3_Ref_Conc']))
RMSE_ER = np.sqrt(np.mean((y_pred - NH3Calc['NH3_Ref_Conc'])**2))
r_ER = np.corrcoef(y_pred, NH3Calc['NH3_Ref_Conc'])[0, 1]
R2_ER = r_ER**2

print('MAE_ER :', MAE_ER )
print('RMSE_ER :', RMSE_ER )
print('r_ER:', r_ER)
print('R^2_ER:', R2_ER)

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Random Forest on original dataset
scaler = StandardScaler()
X = scaler.fit_transform(our_data[['NH3']])
y = reference_data['NH3']

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

y_pred = model.predict(X)

MAE_RF = np.mean(np.abs(y_pred - reference_data['NH3']))
RMSE_RF = np.sqrt(np.mean((y_pred - reference_data['NH3'])**2))
r_RF = np.corrcoef(y_pred, reference_data['NH3'])[0, 1]
R2_RF = r_RF**2

print('MAE_RF :', MAE_RF )
print('RMSE_RF :', RMSE_RF )
print('r_RF:', r_RF)
print('R^2_RF:', R2_RF)

#random forest with new dataframe (without outliers)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(NH3Calc[['NH3_OP_nA']])
y = NH3Calc['NH3_Ref_Conc']

# fit the random forest regression model
model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
model.fit(X, y)
y_pred = model.predict(X)

# calculate metrics for the new model
MAE_RF = np.mean(np.abs(y_pred - NH3Calc['NH3_Ref_Conc']))
RMSE_RF = np.sqrt(np.mean((y_pred - NH3Calc['NH3_Ref_Conc'])**2))
r_RF = np.corrcoef(y_pred, NH3Calc['NH3_Ref_Conc'])[0, 1]
R2_RF = r_RF**2

print('MAE_RF :', MAE_RF )
print('RMSE_RF :', RMSE_RF )
print('r:', r_RF)
print('R^2:', R2_RF)

# tried different max_depth values > 10:0.7, 15:0.79, 20:0.82, 25:0.83, 20 seems to be the optimum

