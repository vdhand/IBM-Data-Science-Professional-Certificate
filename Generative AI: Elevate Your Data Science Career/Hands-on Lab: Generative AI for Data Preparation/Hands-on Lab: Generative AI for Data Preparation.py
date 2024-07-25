#Write a Python code that can perform the following tasks.
#Read the CSV file, located on a given file path, into a Pandas data frame, assuming that the first rows of the file are the headers for the data.

import pandas as pd

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv"
df = pd.read_csv(url)

#Write a Python code that identifies the columns with missing values in a pandas data frame.

# Identify columns with missing values
columns_with_missing_values = df.columns[df.isnull().any()]

#Write a Python code to replace the missing values in a pandas data frame, per the following guidelines.
#1. For a categorical attribute "Screen_Size_cm", replace the missing values with the most frequent value in the column.
#2. For a continuous value attribute "Weight_kg", replace the missing values with the mean value of the entries in the column.

# Replace missing values in the 'Screen_Size_cm' column with the most frequent value
most_frequent_value = df['Screen_Size_cm'].mode()[0]
df['Screen_Size_cm'].fillna(most_frequent_value, inplace=True)
# Replace missing values in the 'Weight_kg' column with the mean value
mean_value = df['Weight_kg'].mean()
df['Weight_kg'].fillna(mean_value, inplace=True)

#Write a Python code snippet to change the data type of the attributes "Screen_Size_cm" and "Weight_kg" of a data frame to float.

# Change the data type of 'Screen_Size_cm' and 'Weight_kg' to float
df['Screen_Size_cm'] = df['Screen_Size_cm'].astype(float)
df['Weight_kg'] = df['Weight_kg'].astype(float)

#Write a Python code to modify the contents under the following attributes of the data frame as required.
#1. Data under 'Screen_Size_cm' is assumed to be in centimeters. Convert this data into inches. Modify the name of the attribute to 'Screen_Size_inch'.
#2. Data under 'Weight_kg' is assumed to be in kilograms. Convert this data into pounds. Modify the name of the attribute to 'Weight_pounds'.

# Convert 'Screen_Size_cm' from centimeters to inches and modify the attribute name
df['Screen_Size_inch'] = df['Screen_Size_cm'] * 0.393701
df.drop('Screen_Size_cm', axis=1, inplace=True)
# Convert 'Weight_kg' from kilograms to pounds and modify the attribute name
df['Weight_pounds'] = df['Weight_kg'] * 2.20462
df.drop('Weight_kg', axis=1, inplace=True)

#Write a Python code to normalize the content under the attribute "CPU_frequency" in a data frame df concerning its maximum value. Make changes to the original data, and do not create a new attribute.

# Normalize the content under 'CPU_frequency' with respect to its maximum value
max_value = df['CPU_frequency'].max()
df['CPU_frequency'] = df['CPU_frequency'] / max_value

#Write a Python code to perform the following tasks.
#1. Convert a data frame df attribute "Screen", into indicator variables, saved as df1, with the naming convention "Screen_<unique value of the attribute>".
#2. Append df1 into the original data frame df.
#3. Drop the original attribute from the data frame df.

# Convert the 'Screen' attribute into indicator variables
df1 = pd.get_dummies(df['Screen'], prefix='Screen')
# Append df1 into the original data frame df
df = pd.concat([df, df1], axis=1)
# Drop the original 'Screen' attribute from the data frame
df.drop('Screen', axis=1, inplace=True)

# Conversion rate from USD to Euro
usd_to_euro_rate = 0.85
# Convert 'Price' values from USD to Euro
df['Price'] = df['Price'] * usd_to_euro_rate

# Perform min-max normalization on the 'CPU_frequency' attribute
min_value = df['CPU_frequency'].min()
max_value = df['CPU_frequency'].max()
df['CPU_frequency'] = (df['CPU_frequency'] - min_value) / (max_value - min_value)

df
