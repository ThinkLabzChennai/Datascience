#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

USAhousing = pd.read _csv('C:/Users/lmohan2/Desktop/New folder/test/USA_Housing.csv')
USAhousing.head()


# In[2]:


USAhousing.info()


# In[3]:


USAhousing.describe()


# In[4]:


USAhousing.columns


# In[5]:


sns.distplot(USAhousing['Price'])


# In[68]:


X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
X.mean()


# In[80]:


X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
X.min()


# In[81]:


X.quantile([0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,.9,.95,1])


# In[88]:


X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
X.max()


# In[57]:


USAhousing.isnull().any()


# In[60]:


X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
l = X.columns.values
number_of_columns=4
number_of_rows = len(l)-1/number_of_columns
plt.figure(figsize=(number_of_columns+10,5*number_of_rows+10))
for i in range(0,len(l)):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.set_style('whitegrid')
    sns.boxplot(X[l[i]],color='orange',orient='v')
    plt.tight_layout()


# In[6]:


sns.heatmap(USAhousing.corr(), annot=True)


# In[7]:


sns.pairplot(USAhousing)


# In[8]:


X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']


# In[9]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[10]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train,y_train)
print(lin_reg.intercept_)
coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
coeff_df


# In[12]:


model = LinearRegression()
results = model.fit(X,y)
print('###  Development data outputs:')
print('R SQUARE:', model.score(X,y)) 


# In[15]:


from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import chi2,f_regression
from sklearn.metrics import mean_squared_error
from math import sqrt
y_pred=model.predict(X)
rms_d = sqrt(mean_squared_error(y,y_pred))
print('RMSE:',rms_d) 
print('\n')


# In[17]:


params = np.append(model.intercept_,model.coef_)
params


# In[28]:


from scipy import stats
def dev_estimates(X,y):
    model = LinearRegression()
    results = model.fit(X,y)
    print('###  Development data outputs:')
    print('R SQUARE:', model.score(X,y)) 
    y_pred=model.predict(X)
    rms_d = sqrt(mean_squared_error(y,y_pred))
    print('RMSE:',rms_d) 
    print('\n')

    params = np.append(model.intercept_,model.coef_)
    y_pred = model.predict(X)

#     newX = pd.DataFrame({"Constant":np.ones(len(X_train))}).join(pd.DataFrame(X_train))
#     MSE = (sum((y_train-y_pred)**2))/(len(newX)-len(newX.columns))

# Note if you don't want to use a DataFrame replace the two lines above with
    newX = np.append(np.ones((len(X),1)), X, axis=1)
    MSE = (sum((y-y_pred)**2))/(len(newX)-len(newX[0]))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]

    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)
    
    col = ['Intercept']
    col.extend(X.columns)
    myDF3 = pd.DataFrame()
    myDF3["Variables"]=col
#     print(col)
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilities"] = [params,sd_b,ts_b,p_values]
    return myDF3
dev_estimates(X_train,y_train)


# In[32]:


def dev_estimates(X,y):
    model = LinearRegression()
    results = model.fit(X_train,y_train)
    print('###  Development data outputs:')
    y_pred=model.predict(X_train)
    params = np.append(model.intercept_,model.coef_)
    print('R SQUARE:', model.score(X_train,y_train)) 
    rms_d = sqrt(mean_squared_error(y_train,y_pred))
    print('RMSE:',rms_d) 
    print('\n')

#     newX = pd.DataFrame({"Constant":np.ones(len(X_train))}).join(pd.DataFrame(X_train))
#     MSE = (sum((y_train-y_pred)**2))/(len(newX)-len(newX.columns))

# Note if you don't want to use a DataFrame replace the two lines above with
    newX = np.append(np.ones((len(X),1)), X, axis=1)
    MSE = (sum((y-y_pred)**2))/(len(newX)-len(newX[0]))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]

    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)
    
    col = ['Intercept']
    col.extend(X.columns)
    myDF3 = pd.DataFrame()
    myDF3["Variables"]=col
#     print(col)
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilities"] = [params,sd_b,ts_b,p_values]
    return myDF3
dev_estimates(X_train,y_train)


# In[35]:


def val_estimate(X,y):
    model = LinearRegression()
    results = model.fit(X_test,y_test)
    print('###  Development data outputs:')
    y_pred=model.predict(X_test)
    params = np.append(model.intercept_,model.coef_)
    print('R SQUARE:', model.score(X_test,y_test)) 
    rms_d = sqrt(mean_squared_error(y_test,y_pred))
    print('RMSE:',rms_d) 
    print('\n')
    
#     newX = pd.DataFrame({"Constant":np.ones(len(X_train))}).join(pd.DataFrame(X_train))
#     MSE = (sum((y_train-y_pred)**2))/(len(newX)-len(newX.columns))

# Note if you don't want to use a DataFrame replace the two lines above with
    newX = np.append(np.ones((len(X),1)), X, axis=1)
    MSE = (sum((y-y_pred)**2))/(len(newX)-len(newX[0]))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]

    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)
    
    col = ['Intercept']
    col.extend(X.columns)
    myDF3 = pd.DataFrame()
    myDF3["Variables"]=col
#     print(col)
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilities"] = [params,sd_b,ts_b,p_values]
    return myDF3
#checking validation estimates
val_estimate(X_test,y_test)


# In[36]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
from scipy import optimize
def dev_VIF(X):
    # VIF dataframe
    X['Constant']=1
    vif_data = pd.DataFrame() 
    vif_data["feature"] = X.columns 
  
    # calculating VIF for each feature 
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(X.shape[1])] 
    X.drop(columns=['Constant'],inplace=True)
  
    return vif_data
dev_VIF(X_train)


# In[38]:


def val_VIF(X):
    # VIF dataframe
    X['Constant']=1
    vif_data = pd.DataFrame() 
    vif_data["feature"] = X.columns 
  
    # calculating VIF for each feature 
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(X.shape[1])] 
    X.drop(columns=['Constant'],inplace=True)
  
    return vif_data
val_VIF(X_test)


# In[40]:


y_pred=model.predict(X_test)
plt.scatter(y_test, y_pred)


# In[41]:


sns.distplot((y_test - y_pred), bins=50);


# In[47]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
final_model = LinearRegression()
results = final_model.fit(X_train,y_train)
y_pred_train=final_model.predict(X_train)
y_pred_test=final_model.predict(X_test)


# In[51]:


import joblib
joblib.dump(final_model, 'model.pkl')


# In[54]:



#Ridge Regression
from sklearn.linear_model import Ridge

model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)


print('R SQUARE:', model.score(X_train,y_train)) 
rms_d = sqrt(mean_squared_error(y_train,train_pred))
print('RMSE:',rms_d) 
print('\n')

print('R SQUARE:', model.score(X_test,y_test)) 
rms_d = sqrt(mean_squared_error(y_test,test_pred))
print('RMSE:',rms_d) 
print('\n')


# In[ ]:


#Ridge Regression parameters

# alpha value - default=1.0
#Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates. 
#Larger values specify stronger regularization.

#solver
#Solver to use in the computational routines:
#‘auto’ chooses the solver automatically based on the type of data.
#‘svd’ uses a Singular Value Decomposition of X to compute the Ridge coefficients. More stable for singular matrices than ‘cholesky’.
#‘cholesky’ uses the standard scipy.linalg.solve function to obtain a closed-form solution.
#‘sparse_cg’ uses the conjugate gradient solver as found in scipy.sparse.linalg.cg. As an iterative algorithm, this solver is more appropriate than ‘cholesky’ for large-scale data (possibility to set tol and max_iter).
#‘lsqr’ uses the dedicated regularized least-squares routine scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative procedure.
#‘sag’ uses a Stochastic Average Gradient descent, and ‘saga’ uses its improved, unbiased version named SAGA. Both methods also use an iterative procedure, and are often faster than other solvers when both n_samples and n_features are large. Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.

#All last five solvers support both dense and sparse data. However, only ‘sag’ and ‘sparse_cg’ supports sparse input when fit_intercept is True.

# RandomState instance, default=None


# In[55]:


#Lasso Regression
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1, 
              precompute=True, 
#               warm_start=True, 
              positive=True, 
              selection='random',
              random_state=42)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)


print('R SQUARE:', model.score(X_train,y_train)) 
rms_d = sqrt(mean_squared_error(y_train,train_pred))
print('RMSE:',rms_d) 
print('\n')

print('R SQUARE:', model.score(X_test,y_test)) 
rms_d = sqrt(mean_squared_error(y_test,test_pred))
print('RMSE:',rms_d) 
print('\n')


# In[ ]:


#precompute
#precomputebool or array-like of shape (n_features, n_features), default=False
#Whether to use a precomputed Gram matrix to speed up calculations. The Gram matrix can also be passed as argument. For sparse input this option is always False to preserve sparsity.

#positivebool, default=False
#When set to True, forces the coefficients to be positive.

#selection{‘cyclic’, ‘random’}, default=’cyclic’
#If set to ‘random’, a random coefficient is updated every iteration rather than looping over features sequentially by default. 
#This (setting to ‘random’) often leads to significantly faster convergence especially when tol is higher than 1e-4.


# In[56]:


#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)

X_train_2_d = poly_reg.fit_transform(X_train)
X_test_2_d = poly_reg.transform(X_test)

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train_2_d,y_train)

test_pred = lin_reg.predict(X_test_2_d)
train_pred = lin_reg.predict(X_train_2_d)

print('R SQUARE:', model.score(X_train,y_train)) 
rms_d = sqrt(mean_squared_error(y_train,train_pred))
print('RMSE:',rms_d) 
print('\n')

print('R SQUARE:', model.score(X_test,y_test)) 
rms_d = sqrt(mean_squared_error(y_test,test_pred))
print('RMSE:',rms_d) 
print('\n')


# In[ ]:




