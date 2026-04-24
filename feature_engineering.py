import pandas as pd
from sklearn.ensemble import RandomForestClassifier

print("Loading data...")
df = pd.read_csv('lending_club_portfolio_data.csv', low_memory=False)

# Keeps only Fully Paid (Good) and Charged Off (Bad) statuses (these are the only loans we want)
valid = ['Fully Paid', 'Charged Off']
df = df[df['loan_status'].isin(valid)].copy()

# Map target to 1 and 0
df['target'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})
df.drop('loan_status', axis=1, inplace=True)

# Dropping all columns that we will not use
columns_to_drop = [
    'id', 'member_id', 'url', 'desc', 'emp_title', 'title', 'zip_code',
    'funded_amnt', 'funded_amnt_inv', 'issue_d', 'out_prncp', 'out_prncp_inv',
    'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
    'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
    'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
    'policy_code', 'pymnt_plan', 'hardship_flag', 'disbursement_method', 'debt_settlement_flag',
    'last_fico_range_high', 'last_fico_range_low'
]
df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

# Dropping columns that have < 50% data
missing_min = df.isnull().sum() / len(df)
columns_to_keep = missing_min[missing_min < 0.50].index
df = df[columns_to_keep]

print(f"Filling in empty data...")

X = df.drop('target', axis=1)
y = df['target']
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Fill in empty data, for numbers use median, for categories label as 'Missing'
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
X[categorical_cols] = X[categorical_cols].fillna('Missing')
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train a  Random Forest to see what it relies on most
print(f"Using random forest classifier to find top 20 features out of {X_encoded.shape[1]}...")
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_encoded, y)

# Extract and rank the "Feature Importances"
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': importances
})

top_20_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(20)
print(f"Top 20: {top_20_features}")

final_columns = top_20_features['Feature'].tolist()
df_final = X_encoded[final_columns].copy()
df_final['target'] = y
df_final.to_csv('lending_club_model_ready.csv', index=False)
print("\nSuccess! Saved 'lending_club_model_ready.csv'. We are ready to build the final model!")