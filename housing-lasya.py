#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv(r"/Users/lasyabuddharaju/Housing.csv")


# In[3]:


df.head(10)


# In[34]:


df_bk=df.copy()


# In[4]:


df.info()


# # data preprocessing
# 

# # 1.data cleaning
# 

# In[5]:


df.isnull().sum()


# # duplicate values

# In[6]:


df_dup=df[df.duplicated(keep='last')]
df_dup


# In[7]:


df=df.drop_duplicates()


# In[8]:


df.nunique()


# # handling noise data

# In[9]:


df[df['price']<0]


# # handling outliers

# In[10]:


from scipy import stats
z_scores = np.abs(stats.zscore(df['price']))
threshold = 3  # Adjust the threshold as needed
outlier_indices = np.where(z_scores > threshold)


# In[11]:


outlier_indices


# In[12]:


df2 = df[z_scores<threshold]
df2


# In[13]:


df['area'].describe()


# In[14]:


sns.histplot(df['area'],kde=True)


# In[15]:


Q1 = df['area'].quantile(0.25)
Q3 = df['area'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['area'] > lower_bound) | (df['area'] < upper_bound)]
df


# # label encoder

# In[16]:


df.info()


# In[17]:


df['mainroad'].value_counts()


# In[18]:


df['mainroad']=df['mainroad'].replace({'yes':0,'no':1})


# In[19]:


df['mainroad'].value_counts()


# In[20]:


df['guestroom'].value_counts()


# In[21]:


df['guestroom']=df['guestroom'].replace({'yes':0,'no':1})


# In[22]:


df['guestroom'].value_counts()


# In[23]:


from sklearn.preprocessing import LabelEncoder
l_impute = LabelEncoder()
df['basement'] = l_impute.fit_transform(df['basement'])
df['hotwaterheating'] = l_impute.fit_transform(df['hotwaterheating'])
df['airconditioning'] = l_impute.fit_transform(df['airconditioning'])
df['prefarea'] = l_impute.fit_transform(df['prefarea'])
df['furnishingstatus'] = l_impute.fit_transform(df['furnishingstatus'])


# In[24]:


df['basement'].value_counts()


# In[25]:


df


# In[26]:


df.isnull().sum()


# # Normalization

# In[27]:


Indepvar=[]
for col in df.columns:
    if col !='price':
        Indepvar.append(col)
Targetvar ='price'

x= df[Indepvar]
y=df[Targetvar]


# In[28]:


x.head()


# In[29]:


y.head()


# In[30]:


#split data into train and test (random sampling)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#displey the shape of train and test data

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[31]:


#scaling is done to bring all the varaibles on smiliar scale. 

from sklearn.preprocessing import MinMaxScaler

mmscaler=MinMaxScaler(feature_range=(0,1))

x_train = mmscaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train)

x_test = mmscaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test)


# # Linear Regression

# In[32]:


# Build the multi regression model

from sklearn.linear_model import LinearRegression

# Create object for the model

ModelMLR = LinearRegression()

# Train the model with training data

ModelMLR.fit(x_train, y_train)

# Predict the model with test dataset

y_pred = ModelMLR.predict(x_test)

# Evaluation metrics for Regression analysis

from sklearn import metrics

print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, y_pred),3))
print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, y_pred),3))
print('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),3))
print('R2_score:', round(metrics.r2_score(y_test, y_pred),6))
print('Root Mean Squared Log Error (RMSLE):', round(np.log(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),3))
print('Mean Absolute Percentage Error (MAPE):', round(metrics.mean_absolute_percentage_error(y_test, y_pred)*100,3), '%')
# Define the function to calculate the MAPE - Mean Absolute Percentage Error

def MAPE (y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Evaluation of MAPE

result = MAPE(y_test, y_pred)
print('Mean Absolute Percentage Error (MAPE):', round(result, 3), '%')
# Calculate Adjusted R squared values

r_squared = round(metrics.r2_score(y_test, y_pred),6)
adjusted_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1),6)
print('Adj R Square: ', adjusted_r_squared)


# In[67]:


# Display the Final results

Results = pd.DataFrame({'price_A':y_test, 'price_P':y_pred})

# Merge two Dataframes on index of both the dataframes

ResultsFinal = df_bk.merge(Results, left_index=True, right_index=True)
#calculate the % of error

ResultsFinal['%Error']=round(((ResultsFinal['price_A']-ResultsFinal['price_P'])/ResultsFinal['price_A'])*100,3)
# Display 10 records randomly

ResultsFinal.sample(5)


# In[68]:


# Build the Regression / Regressor models

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR

# Create objects of Regression / Regressor models with default hyper-parameters

ModelMLR = LinearRegression()
ModelDCR = DecisionTreeRegressor()
ModelRFR = RandomForestRegressor()
ModelETR = ExtraTreesRegressor()
ModelKNN = KNeighborsRegressor(n_neighbors=5)
ModelBRR = BayesianRidge()
ModelSVR = SVR()

# Evalution matrix for all the algorithms

MM = [ModelMLR, ModelDCR, ModelRFR, ModelETR, ModelKNN, ModelBRR, ModelSVR]
#MM = [ModelMLR, ModelDCR, ModelRFR, ModelETR]

for models in MM:

    # Fit the model with train data

    models.fit(x_train, y_train)

    # Predict the model with test data

    y_pred = models.predict(x_test)

    # Print the model name

    print('Model Name: ', models)

    # Evaluation metrics for Regression analysis

    from sklearn import metrics

    print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, y_pred),3))
    print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, y_pred),3))
    print('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),3))
    print('R2_score:', round(metrics.r2_score(y_test, y_pred),6))
    print('Root Mean Squared Log Error (RMSLE):', round(np.log(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),3))

    # Define the function to calculate the MAPE - Mean Absolute Percentage Error

    def MAPE (y_test, y_pred):
        y_test, y_pred = np.array(y_test), np.array(y_pred)
        return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Evaluation of MAPE

    result = MAPE(y_test, y_pred)
    print('Mean Absolute Percentage Error (MAPE):', round(result, 2), '%')

    # Calculate Adjusted R squared values

    r_squared = round(metrics.r2_score(y_test, y_pred),6)
    adjusted_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1),6)
    print('Adj R Square: ', adjusted_r_squared)
    print('------------------------------------------------------------------------------------------------------------')
    #-------------------------------------------------------------------------------------------
    new_row = {'Model Name' : models,
               'Mean_Absolute_Error_MAE' : metrics.mean_absolute_error(y_test, y_pred),
               'Adj_R_Square' : adjusted_r_squared,
               'Root_Mean_Squared_Error_RMSE' : np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
               'Mean_Absolute_Percentage_Error_MAPE' : result,
               'Mean_Squared_Error_MSE' : metrics.mean_squared_error(y_test, y_pred),
               'Root_Mean_Squared_Log_Error_RMSLE': np.log(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),
               'R2_score' : metrics.r2_score(y_test, y_pred)}


# In[69]:


# Predict the values with DecisionTree algorithm

y_predF=ModelKNN.predict(x_test)


# In[70]:


# Display the final results

Results=pd.DataFrame({'Price_A':y_test,'Price_P':y_predF})

# Merge two dataframes on index of both dataframes

ResultsFinal=df.merge(Results,left_index=True,right_index=True)

# Display 5 records randomly

ResultsFinal.sample(5)


# In[71]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Generate sample data (replace this with your actual data)
X, y = make_regression(n_samples=100, n_features=1, noise=0.2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the K Neighbors Regressor
k = 5  # Number of neighbors
knn_regressor = KNeighborsRegressor(n_neighbors=k)
knn_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = knn_regressor.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.title('K Neighbors Regressor: Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


# In[72]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Generate sample data (replace this with your actual data)
X, y = make_regression(n_samples=100, n_features=1, noise=0.2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Random Forest Regressor
random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = random_forest_regressor.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.title('Random Forest Regressor: Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


# In[ ]:




