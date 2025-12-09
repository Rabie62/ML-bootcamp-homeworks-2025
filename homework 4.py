import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('course_lead_scoring.csv')

# Identify categorical and numerical columns
categorical = ['lead_source', 'industry', 'employment_status', 'location']
numerical = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score']
target = 'converted'

# Check for missing values
print("Missing values per column:")
print(df.isna().sum())

# Fill missing values
for col in categorical:
    df[col] = df[col].fillna('NA')

for col in numerical:
    df[col] = df[col].fillna(0.0)

# Cap outliers using IQR method
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower_bound, upper_bound)
    return df

for col in numerical:
    df = cap_outliers(df, col)

# Split the data: 60% train, 20% val, 20% test
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)

# Extract target variables before dropping
y_train = df_train[target].values
y_val = df_val[target].values
y_test = df_test[target].values
y_train_full = df_train_full[target].values

# Drop target column from datasets
df_train = df_train.drop(columns=[target])
df_val = df_val.drop(columns=[target])
df_test = df_test.drop(columns=[target])
df_train_full = df_train_full.drop(columns=[target])

# Scale numerical features
scaler = StandardScaler()
df_train[numerical] = scaler.fit_transform(df_train[numerical])
df_val[numerical] = scaler.transform(df_val[numerical])
df_test[numerical] = scaler.transform(df_test[numerical])

# Combine train and val for full training set
df_full_train = pd.concat([df_train, df_val])
y_full_train = np.concatenate([y_train, y_val])

# Question 1: ROC AUC feature importance
auc_scores = {}
for col in numerical:
    auc = roc_auc_score(y_train, df_train[col])
    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[col])
    auc_scores[col] = auc

best_feature = max(auc_scores, key=auc_scores.get)
print(f"\nQuestion 1: The numerical variable with the highest AUC is {best_feature}")

# Question 2: Training the model with one-hot encoding
features = categorical + numerical

dv = DictVectorizer(sparse=False)

train_dicts = df_train[features].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

val_dicts = df_val[features].to_dict(orient='records')
X_val = dv.transform(val_dicts)

y_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print(f"\nQuestion 2: AUC on validation dataset is {round(auc, 3)}")

# Question 3: Precision and Recall
thresholds = np.arange(0.0, 1.01, 0.01)
precisions = []
recalls = []

actual_positive = (y_val == 1)
actual_negative = (y_val == 0)

for t in thresholds:
    predicted_positive = (y_pred >= t)
    tp = (predicted_positive & actual_positive).sum()
    fp = (predicted_positive & actual_negative).sum()
    fn = (~predicted_positive & actual_positive).sum()
    
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    precisions.append(p)
    recalls.append(r)

plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.show()

intersections = [t for i, t in enumerate(thresholds) if abs(precisions[i] - recalls[i]) < 0.01]
intersection_threshold = intersections[0] if intersections else None
print(f"\nQuestion 3: Precision and recall curves intersect at threshold approximately {round(intersection_threshold, 3)}")

# Question 4: F1 score
f1_scores = []
for i in range(len(thresholds)):
    p = precisions[i]
    r = recalls[i]
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    f1_scores.append(f1)

max_f1_idx = np.argmax(f1_scores)
max_f1_threshold = thresholds[max_f1_idx]
print(f"\nQuestion 4: Threshold where F1 is maximal is {round(max_f1_threshold, 2)}")

# Question 5: 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []

for train_idx, val_idx in kf.split(df_full_train):
    df_train_fold = df_full_train.iloc[train_idx]
    df_val_fold = df_full_train.iloc[val_idx]
    
    y_train_fold = y_full_train[train_idx]
    y_val_fold = y_full_train[val_idx]
    
    train_dicts_fold = df_train_fold[features].to_dict(orient='records')
    X_train_fold = dv.fit_transform(train_dicts_fold)
    
    model_fold = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model_fold.fit(X_train_fold, y_train_fold)
    
    val_dicts_fold = df_val_fold[features].to_dict(orient='records')
    X_val_fold = dv.transform(val_dicts_fold)
    
    y_pred_fold = model_fold.predict_proba(X_val_fold)[:, 1]
    auc_fold = roc_auc_score(y_val_fold, y_pred_fold)
    scores.append(auc_fold)

std_dev = np.std(scores)
print(f"\nQuestion 5: Standard deviation of the scores across different folds is {round(std_dev, 3)}")

# Question 6: Hyperparameter Tuning
C_values = [0.000001, 0.001, 1]
results = {}

for C in C_values:
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []
    
    for train_idx, val_idx in kf.split(df_full_train):
        df_train_fold = df_full_train.iloc[train_idx]
        df_val_fold = df_full_train.iloc[val_idx]
        
        y_train_fold = y_full_train[train_idx]
        y_val_fold = y_full_train[val_idx]
        
        train_dicts_fold = df_train_fold[features].to_dict(orient='records')
        X_train_fold = dv.fit_transform(train_dicts_fold)
        
        model_fold = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model_fold.fit(X_train_fold, y_train_fold)
        
        val_dicts_fold = df_val_fold[features].to_dict(orient='records')
        X_val_fold = dv.transform(val_dicts_fold)
        
        y_pred_fold = model_fold.predict_proba(X_val_fold)[:, 1]
        auc_fold = roc_auc_score(y_val_fold, y_pred_fold)
        scores.append(auc_fold)
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    results[C] = (round(mean_score, 3), round(std_score, 3))

best_C = min([(C, results[C][0], results[C][1]) for C in C_values], key=lambda x: (-x[1], x[2], x[0]))[0]
print(f"\nQuestion 6: C that leads to the best mean score is {best_C}")