import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_splitA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import mean_absolute_error
housing = pd.read_csv('Housing.csv')
varlist = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
def binary_map(x):
    return x.map({'yes': 1, "no": 0})
housing[varlist] = housing[varlist].apply(binary_map)
status = pd.get_dummies(housing['furnishingstatus'])
status.head()
#we can use 2 columns to identify 3 things
# 00  furnished
# 10 - semi-furnishes
# 01 - unfurnished

#lets drop 1st columnn
status = pd.get_dummies(housing['furnishingstatus'], drop_first = True)S
housing = pd.concat([housing, status], axis = 1)
housing.head()
housing.drop(['furnishingstatus'], axis = 1, inplace = True)
housing.head()
# Split the data into training and testing sets
train_data, test_data = train_test_split(housing, test_size=0.2, random_state=42)
train_data
train_data.columns
train_data.isnull().sum()
test_data.isnull().sum()
#initial hypothese variables = 'area', 'bedrooms',' mainroad', 'prefare', 'parking', '

plt.scatter(train_data['area'], train_data['price'])
plt.xlabel('area')
plt.ylabel('price')
plt.show()
plt.bar(train_data['bedrooms'], train_data['price'])
plt.xlabel('bedrooms')
plt.ylabel('price')
plt.show()
plt.bar(train_data['mainroad'], train_data['price'])
plt.xlabel('mainroad')
plt.ylabel('price')
plt.show()
plt.bar(train_data['prefarea'], train_data['price'])
plt.xlabel('prefare')
plt.ylabel('price')
plt.show()
plt.bar(train_data['parking'], train_data['price'])
plt.xlabel('parking')
plt.ylabel('price')
plt.show()
#creating correlation matrix to understand how each variable is realed to target variable
corr_matrix = train_data.corr()
print(corr_matrix)
x_train = train_data[['area', 'stories', 'bathrooms', 'airconditioning']]
y_train = train_data['price']

model1 = LinearRegression()
score1 = cross_val_score(model1, x_train, y_train, cv = 10, scoring = 'r2')
score2 = cross_val_score(model1, x_train, y_train, cv = 10, scoring = 'neg_root_mean_squared_error')
print("The RMSE value is ", np.abs(score2.mean()))
print("The R-squared value is ", score1.mean())

x_train = train_data[['area', 'stories', 'bathrooms', 'airconditioning','parking']]
y_train = train_data['price']

model2 = LinearRegression()
score1 = cross_val_score(model2, x_train, y_train, cv = 10, scoring = 'r2')
score2 = cross_val_score(model2, x_train, y_train, cv = 10, scoring = 'neg_root_mean_squared_error')
print("The RMSE value is ", np.abs(score2.mean()))
print("The R-squared value is ", score1.mean())

x_train = train_data[['area', 'stories', 'bathrooms', 'airconditioning','mainroad']]
y_train = train_data['price']

model1 = LinearRegression()
score1 = cross_val_score(model1, x_train, y_train, cv = 10, scoring = 'r2')
score2 = cross_val_score(model1, x_train, y_train, cv = 10, scoring = 'neg_root_mean_squared_error')
print("The RMSE value is ", np.abs(score2.mean()))
print("The R-squared value is ", score1.mean())

train_data.hist(column='price')

train_data['logged_price']= np.log(train_data.price)
train_data.head()
train_data.hist(column = 'logged_price')
train_data.hist(column = 'area')
train_data['logged_area']= np.log(train_data.price)
train_data.hist(column = 'logged_area')
train_data.hist(column = 'bathrooms')
train_data['logged_bathrooms']= np.log(train_data.price)
train_data.hist(column = 'logged_bathrooms')
train_data.hist(column='stories')
train_data['logged_stories']= np.log(train_data.price)
train_data.hist(column='logged_stories')
train_data.hist(column='mainroad')
train_data['logged_mainroad']= np.log(train_data.price)
train_data.hist(column='logged_mainroad')
train_data.hist(column='guestroom')
train_data['logged_guestroom']= np.log(train_data.price)
train_data.hist(column='logged_guestroom')
train_data.hist(column='basement')
train_data['logged_basement']= np.log(train_data.price)
train_data.hist(column='logged_basement')
train_data.hist(column='hotwaterheating')
train_data['logged_hotwaterheating']= np.log(train_data.price)
train_data.hist(column='logged_hotwaterheating')
train_data.hist(column='airconditioning')
train_data['logged_airconditioning']= np.log(train_data.price)
train_data.hist(column='logged_airconditioning')
train_data.hist(column='parking')
train_data['logged_parking']= np.log(train_data.price)
train_data.hist(column='logged_parking')
train_data.hist(column='prefarea')
train_data['logged_prefarea']= np.log(train_data.price)
train_data.hist(column='logged_prefarea')
train_data.hist(column='semi-furnished')
train_data['logged_semi-furnished']= np.log(train_data.price)
train_data.hist(column='logged_semi-furnished')
x_train = train_data[['area','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea', 'semi-furnished']]
y_train = train_data['logged_price']

model4 = LinearRegression()
score1 = cross_val_score(model4, x_train, y_train, cv = 10, scoring = 'r2')
score2 = cross_val_score(model4, x_train, y_train, cv = 10, scoring = 'neg_root_mean_squared_error')
print("The RMSE value is ", np.abs(score2.mean()))
print("The R-squared value is ", score1.mean())
model4.fit(x_train, y_train)

# Make predictions on the training data
y_train_pred = model4.predict(x_train)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_train, y_train_pred)
print("The MAE value is", mae)

model4.fit(x_train, y_train)

# Make predictions on the training data
y_train_pred = model4.predict(x_train)

# Create a scatter plot for actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.xlabel("Actual Logged Price")
plt.ylabel("Predicted Logged Price")
plt.title("Actual vs. Predicted Values")

# Add a regression line (best fit line)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', lw=2)
plt.show()












