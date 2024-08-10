'''Implement a linear regression model to predict the prices 
of houses based on their square footage and the number of bedrooms and bathrooms.'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('train.csv')
prin(df)

# Select the relevant features
x = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred,squared=False)
print(f"Mean Squared Error: {mse:.2f}")

#prediction for a new house
new_home1= pd.DataFrame([[2100,3,2]], columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])
predict_price1= model.predict(new_home1)
print(f"Predicted price of new home: ${predict_price1[0]:,.2f}")

new_home2= pd.DataFrame([[3000,3,3]], columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])
predict_price2= model.predict(new_home2)
print(f"Predicted price of new home: ${predict_price2[0]:,.2f}")
