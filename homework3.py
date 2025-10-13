import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif

# Load the dataset
df = pd.read_csv('course_lead_scoring.csv')

# Question 1: Data Preparation and Mode for 'industry'
categorical_cols = ['lead_source', 'industry', 'employment_status', 'location']
numerical_cols = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score']

for col in categorical_cols:
    df[col] = df[col].fillna('NA')

for col in numerical_cols:
    df[col] = df[col].fillna(0.0)

# Most frequent observation (mode) for 'industry'
industry_mode = df['industry'].mode()[0]
print(f"Question 1: Most frequent industry: {industry_mode}")

# Question 2: Correlation matrix for numerical features
correlation_matrix = df[numerical_cols].corr()
pairs = [
    ('interaction_count', 'lead_score'),
    ('number_of_courses_viewed', 'lead_score'),
    ('number_of_courses_viewed', 'interaction_count'),
    ('annual_income', 'interaction_count')
]
max_corr = -1
max_pair = None
for col1, col2 in pairs:
    corr = abs(correlation_matrix.loc[col1, col2])
    if corr > max_corr:
        max_corr = corr
        max_pair = (col1, col2)
print(f"Question 2: Features with biggest correlation: {max_pair[0]} and {max_pair[1]}")

# Question 3: Split data and calculate mutual information
X = df.drop(columns=['converted'])
y = df['converted']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Encode categorical variables for mutual information
categorical_features = ['industry', 'location', 'lead_source', 'employment_status']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat_encoded = encoder.fit_transform(X_train[categorical_features])

# Mutual information for categorical variables
mi_scores = mutual_info_classif(
    X_train_cat_encoded, y_train, discrete_features=True, random_state=42
)

# Sum MI scores for each original feature (since one-hot encoding creates multiple columns)
mi_scores_dict = {}
for feature in categorical_features:
    feature_cols = [col for col in encoder.get_feature_names_out() if col.startswith(feature)]
    feature_indices = [i for i, col in enumerate(encoder.get_feature_names_out()) if col.startswith(feature)]
    mi_scores_dict[feature] = round(sum(mi_scores[indices] for indices in feature_indices), 2)

max_mi_feature = max(mi_scores_dict, key=mi_scores_dict.get)
print(f"Question 3: Variable with biggest MI score: {max_mi_feature} ({mi_scores_dict[max_mi_feature]})")

# Question 4: Logistic Regression with One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_cols)
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)
val_accuracy = accuracy_score(y_val, model.predict(X_val))
val_accuracy_rounded = round(val_accuracy, 2)
print(f"Question 4: Validation accuracy: {val_accuracy_rounded}")

# Question 5: Feature Elimination
original_accuracy = val_accuracy
features_to_test = ['industry', 'employment_status', 'lead_score']
accuracy_differences = {}

for feature in features_to_test:
    if feature in categorical_features:
        temp_cols = [col for col in categorical_features if col != feature] + numerical_cols
    else:
        temp_cols = categorical_features + [col for col in numerical_cols if col != feature]
    
    temp_preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), 
             [col for col in temp_cols if col in categorical_features]),
            ('num', 'passthrough', [col for col in temp_cols if col in numerical_cols])
        ])
    
    temp_model = Pipeline([
        ('preprocessor', temp_preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42))
    ])
    
    temp_model.fit(X_train[temp_cols], y_train)
    temp_accuracy = accuracy_score(y_val, temp_model.predict(X_val[temp_cols]))
    accuracy_differences[feature] = original_accuracy - temp_accuracy

min_diff_feature = min(accuracy_differences, key=lambda x: abs(accuracy_differences[x]))
print(f"Question 5: Feature with smallest difference: {min_diff_feature}")

# Question 6: Regularized Logistic Regression with different C values
C_values = [0.01, 0.1, 1, 10, 100]
accuracies = {}

for C in C_values:
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)
    acc = accuracy_score(y_val, model.predict(X_val))
    accuracies[C] = round(acc, 3)

best_C = min([C for C in C_values if accuracies[C] == max(accuracies.values())])
print(f"Question 6: Best C value: {best_C}")