import pandas as pd

# Specify the file path of the CSV file
file_path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv'

#Write a python code to perform the following actions.
#1. Import a data set from a CSV file, The headers for the data set must be in the first row of the CSV file.
#2. Generate the statistical description of all the features used in the data set. Include "object" data types as well.

# Import data set from a CSV file
data = pd.read_csv('file_path')

# Generate statistical description of all features
description = data.describe(include='all')

#Write a Python code to perform the following actions.
#1. Create regression plots for the attributes "CPU_frequency", "Screen_Size_inch" and "Weight_pounds" against "Price".
#2. Create box plots for the attributes "Category", "GPU", "OS", "CPU_core", "RAM_GB" and "Storage_GB_SSD" against the attribute "Price".


import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

# Assuming 'data' is your DataFrame with the required attributes

# Create regression plots
sns.pairplot(data, x_vars=["CPU_frequency", "Screen_Size_inch", "Weight_pounds"], y_vars=["Price"], kind="reg")
plt.show()

# Create box plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
sns.boxplot(x="Category", y="Price", data=data, ax=axes[0, 0])
sns.boxplot(x="GPU", y="Price", data=data, ax=axes[0, 1])
sns.boxplot(x="OS", y="Price", data=data, ax=axes[0, 2])
sns.boxplot(x="CPU_core", y="Price", data=data, ax=axes[1, 0])
sns.boxplot(x="RAM_GB", y="Price", data=data, ax=axes[1, 1])
sns.boxplot(x="Storage_GB_SSD", y="Price", data=data, ax=axes[1, 2])
plt.tight_layout()
plt.show()

#Write a Python code for the following.
#1. Evaluate the correlation value, pearson coefficient and p-values for all numerical attributes against the target attribute "Price".
#2. Don't include the values evaluated for target variable against itself.
#3. Print these values as a part of a single dataframe against each individual attrubute.

# Evaluate correlation, Pearson coefficient, and p-values
correlation_values = []
pearson_coefficients = []
p_values = []

numerical_attributes = data.select_dtypes(include=[np.number]).columns

for attribute in numerical_attributes:
    if attribute != "Price":
        correlation = data[attribute].corr(data["Price"])
        pearson_coef, p_value = stats.pearsonr(data[attribute], data["Price"])
        correlation_values.append(correlation)
        pearson_coefficients.append(pearson_coef)
        p_values.append(p_value)

# Create a DataFrame with the evaluated values
results_df = pd.DataFrame({
    'Attribute': numerical_attributes[numerical_attributes != "Price"],
    'Correlation': correlation_values,
    'Pearson Coefficient': pearson_coefficients,
    'P-Value': p_values
})

print(results_df)

#Write a python code that performs the following actions.
#1. Group the attributes "GPU", "CPU_core" and "Price", as available in a dataframe df
#2. Create a pivot table for this group, assuming the target variable to be 'Price' and aggregation function as mean
#3. Plot a pcolor plot for this pivot table.

# Group the attributes
grouped_df = df.groupby(['GPU', 'CPU_core'])['Price'].mean().reset_index()

# Create a pivot table
pivot_table = grouped_df.pivot(index='GPU', columns='CPU_core', values='Price')

# Plot a pcolor plot
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap='coolwarm')
plt.title('Pivot Table for GPU, CPU_core, and Price')
plt.xlabel('CPU_core')
plt.ylabel('GPU')
plt.show()
