
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('car_fuel_efficiency.csv')

print(pd.__version__)

# Q2. Records count
print(f"Q2. Number of records: {len(df)}")

# Q3. Fuel types
print(f"Q3. Number of fuel types: {df['fuel_type'].nunique()}")

# Q4. Missing values
missing_cols = df.isnull().sum()
missing_cols = missing_cols[missing_cols > 0]
print(f"Q4. Number of columns with missing values: {len(missing_cols)}")

# Q5. Max fuel efficiency for cars from Asia
asia_max_mpg = df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max()
print(f"Q5. Max fuel efficiency for Asia: {asia_max_mpg:.2f}")

# Q6. Median value of horsepower analysis
# 1. Find the initial median value of horsepower
initial_median = df['horsepower'].median()
print(f"Q6.1. Initial median horsepower: {initial_median}")

# 2. Calculate the most frequent value of horsepower
most_frequent_hp = df['horsepower'].mode()[0]
print(f"Q6.2. Most frequent horsepower: {most_frequent_hp}")

# 3. Fill missing values with the most frequent value
df_filled = df.copy()
df_filled['horsepower'] = df_filled['horsepower'].fillna(most_frequent_hp)

# 4. Calculate the median value again
new_median = df_filled['horsepower'].median()
print(f"Q6.4. New median horsepower: {new_median}")
print(f"Q6.4. Has it changed? {'Yes' if initial_median != new_median else 'No'}")

# Q7. Sum of weights
# 1. Select all the cars from Asia
asia_cars = df[df['origin'] == 'Asia']

# 2. Select only columns vehicle_weight and model_year
X_df = asia_cars[['vehicle_weight', 'model_year']]

# 3. Select the first 7 values
X_df = X_df.head(7)

# 4. Get the underlying NumPy array
X = X_df.values

# 5. Compute matrix-matrix multiplication between the transpose of X and X
XTX = X.T.dot(X)

# 6. Invert XTX
XTX_inv = np.linalg.inv(XTX)

# 7. Create array y
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

# 8. Multiply the inverse of XTX with the transpose of X, then multiply by y
w = XTX_inv.dot(X.T).dot(y)

# 9. Sum of all elements
w_sum = np.sum(w)
print(f"Q7. Sum of all elements of w: {w_sum}")
