#Write a Python code that can perform the following tasks.
#Read the CSV file, located on a given file path, into a pandas data frame, assuming that the first row of the file can be used as the headers for the data.

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Specify the file path where the CSV file is located
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"

# Read the CSV file into a pandas data frame
df = pd.read_csv(file_path)

# Display the data frame
print(df)

#Write a Python code that performs the following tasks.
#1. Develops and trains a linear regression model that uses one attribute of a data frame as the source variable and another as a target variable.
#2. Calculate and display the MSE and R^2 values for the trained model

source_column = 'CPU_frequency'
target_column = 'Price'

# Split the data into source and target variables
X = df[[source_column]]
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared (R^2) values
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the MSE and R-squared values
print("Mean Squared Error (MSE):", mse)
print("R-squared (R^2):", r2)

#Write a Python code that performs the following tasks.
#1. Develops and trains a linear regression model that uses some attributes of a data frame as the source variables and one of the attributes as a target variable.
#2. Calculate and display the MSE and R^2 values for the trained model.

# Define the list of source columns and the target column
source_columns = ['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']
target_column = 'Price'

# Split the data into source and target variables
X = df[source_columns]
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared (R^2) values
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the MSE and R-squared values
print("Mean Squared Error (MSE):", mse)
print("R-squared (R^2):", r2)

#Write a Python code that performs the following tasks.
#1. Develops and trains multiple polynomial regression models, with orders 2, 3, and 5, that use one attribute of a data frame as the source variable and another as a target variable.
#2. Calculate and display the MSE and R^2 values for the trained models.
#3. Compare the performance of the models.

# Define the source and target columns
source_column = 'CPU_frequency'
target_column = 'Price'

# Split the data into source and target variables
X = df[[source_column]]
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features for different orders
orders = [2, 3, 5]
for order in orders:
    poly_features = PolynomialFeatures(degree=order)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # Create and train the polynomial regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Make predictions
    y_pred = model.predict(X_test_poly)

    # Calculate the Mean Squared Error (MSE) and R-squared (R^2) values
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display the MSE and R-squared values for the current model
    print(f"Polynomial Regression Model with order {order}:")
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R^2):", r2)
    print()

#Write a Python code that performs the following tasks.
#1. Create a pipeline that performs parameter scaling, Polynomial Feature generation, and Linear regression. Use the set of multiple features as before to create this pipeline.
#2. Calculate and display the MSE and R^2 values for the trained model.

# Define the list of source columns and the target column
source_columns = ['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']
target_column = 'Price'

# Split the data into source and target variables
X = df[source_columns]
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for parameter scaling, polynomial feature generation, and linear regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly_features', PolynomialFeatures(degree=2)),
    ('linear_regression', LinearRegression())
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared (R^2) values
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the MSE and R-squared values
print("Mean Squared Error (MSE):", mse)
print("R-squared (R^2):", r2)

#Write a Python code that performs the following tasks.
#1. Use polynomial features for some of the attributes of a data frame.
#2. Perform Grid search on a ridge regression model for a set of values of hyperparameter alpha and polynomial features as input.
#3. Use cross-validation in the Grid search.
#4. Evaluate the resulting model's MSE and R^2 values.

source_columns = ['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']
target_column = 'Price'
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10]
poly_order = 2

# Create polynomial features for the selected attributes
poly_features = PolynomialFeatures(degree=poly_order)
X_poly = poly_features.fit_transform(df[source_columns])
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Perform Grid Search with Ridge Regression
param_grid = {'alpha': alpha_values}
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=4)
grid_search.fit(X_train, y_train)

# Evaluate the best model from Grid Search
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared (R^2) values
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the MSE and R-squared values
print("Mean Squared Error (MSE):", mse)
print("R-squared (R^2):", r2)


