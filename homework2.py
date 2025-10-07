import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('/kaggle/input/cars-dataset/car_fuel_efficiency.csv')

# Select only the required columns
df_filtered = df[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']]

print(df_filtered['fuel_efficiency_mpg'].describe())

plt.hist(df_filtered['fuel_efficiency_mpg'], bins=30)
plt.title('Distribution of fuel_efficiency_mpg')
plt.xlabel('Fuel Efficiency (MPG)')
plt.ylabel('Frequency')
plt.show()

# Question 1: Identify column with missing values
missing_values = df_filtered.isnull().sum()
print("\nMissing values:")
print(missing_values)

# Question 2: Compute median of horsepower
median_horsepower = df_filtered['horsepower'].median()
print("\nMedian horsepower:", median_horsepower)

# Prepare and split the dataset
np.random.seed(42)  
df_shuffled = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(df_shuffled)
n_train = int(0.6 * n)
n_val = int(0.2 * n)
n_test = n - n_train - n_val

df_train = df_shuffled.iloc[:n_train]
df_val = df_shuffled.iloc[n_train:n_train + n_val]
df_test = df_shuffled.iloc[n_train + n_val:]

# Question 3: Train linear regression with missing values filled with 0 and mean
X_train = df_train[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].copy()
y_train = df_train['fuel_efficiency_mpg']
X_val = df_val[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].copy()
y_val = df_val['fuel_efficiency_mpg']

# Option 1: Fill missing values with 0
X_train_zero = X_train.fillna(0)
X_val_zero = X_val.fillna(0)

model_zero = LinearRegression()
model_zero.fit(X_train_zero, y_train)
y_pred_zero = model_zero.predict(X_val_zero)
rmse_zero = round(np.sqrt(mean_squared_error(y_val, y_pred_zero)), 2)
print("\nRMSE with missing values filled with 0:", rmse_zero)

# Option 2: Fill missing values with mean 
horsepower_mean = X_train['horsepower'].mean()
X_train_mean = X_train.fillna({'horsepower': horsepower_mean})
X_val_mean = X_val.fillna({'horsepower': horsepower_mean})

model_mean = LinearRegression()
model_mean.fit(X_train_mean, y_train)
y_pred_mean = model_mean.predict(X_val_mean)
rmse_mean = round(np.sqrt(mean_squared_error(y_val, y_pred_mean)), 2)
print("\nRMSE with missing values filled with mean:", rmse_mean)

print("\nQuestion 3 Answer: ", "With 0" if rmse_zero < rmse_mean else "With mean" if rmse_mean < rmse_zero else "Both are equally good")

# Question 4: Train regularized linear regression with different r values
from sklearn.linear_model import Ridge

r_values = [0, 0.01, 0.1, 1, 5, 10, 100]
rmse_scores = []

# Use fill with 0 as specified
X_train_zero = X_train.fillna(0)
X_val_zero = X_val.fillna(0)

for r in r_values:
    model_ridge = Ridge(alpha=r)
    model_ridge.fit(X_train_zero, y_train)
    y_pred_ridge = model_ridge.predict(X_val_zero)
    rmse = round(np.sqrt(mean_squared_error(y_val, y_pred_ridge)), 2)
    rmse_scores.append((r, rmse))
    print(f"\nRMSE with r={r}: {rmse}")

# Find the best r 
best_r, best_rmse = min(rmse_scores, key=lambda x: (x[1], x[0]))
print(f"\nQuestion 4 Answer: Best r = {best_r} with RMSE = {best_rmse}")

# Question 5: Evaluate model stability with different seeds
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rmse_scores_seeds = []

for seed in seeds:
    np.random.seed(seed)
    df_shuffled = df_filtered.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Split data
    df_train = df_shuffled.iloc[:n_train]
    df_val = df_shuffled.iloc[n_train:n_train + n_val]
    df_test = df_shuffled.iloc[n_train + n_val:]
    
    X_train = df_train[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].fillna(0)
    y_train = df_train['fuel_efficiency_mpg']
    X_val = df_val[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].fillna(0)
    y_val = df_val['fuel_efficiency_mpg']
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores_seeds.append(rmse)
    print(f"\nRMSE with seed {seed}: {rmse}")

std_rmse = round(np.std(rmse_scores_seeds), 3)
print(f"\nQuestion 5 Answer: Standard deviation of RMSE scores = {std_rmse}")

# Question 6: Train model with seed 9, combine train+val, r=0.001, evaluate on test
np.random.seed(9)
df_shuffled = df_filtered.sample(frac=1, random_state=9).reset_index(drop=True)

# Split data
df_train = df_shuffled.iloc[:n_train]
df_val = df_shuffled.iloc[n_train:n_train + n_val]
df_test = df_shuffled.iloc[n_train + n_val:]

# Combine train and validation
df_train_val = pd.concat([df_train, df_val]).reset_index(drop=True)

X_train_val = df_train_val[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].fillna(0)
y_train_val = df_train_val['fuel_efficiency_mpg']
X_test = df_test[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].fillna(0)
y_test = df_test['fuel_efficiency_mpg']

# Train Ridge model with r=0.001
model_final = Ridge(alpha=0.001)
model_final.fit(X_train_val, y_train_val)
y_pred_test = model_final.predict(X_test)
rmse_test = round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 3)
print(f"\nQuestion 6: RMSE on test set with seed 9 and r=0.001: {rmse_test}")



    