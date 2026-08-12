import kagglehub
import pandas as pd
import xgboost as xgb
import joblib
import os

# 1. Download latest version of the dataset from Kaggle
path = kagglehub.dataset_download("mirichoi0218/insurance")
print("Path to dataset files:", path)

# 2. Find and load the dataset file from the downloaded path
files = os.listdir(path)
data_file = [f for f in files if f.endswith('.csv') or f.endswith('.xlsx')][0]
full_path = os.path.join(path, data_file)

df = pd.read_csv(full_path)


# Check if user submissions exist and merge them
if os.path.exists('user_submissions.csv'):
    user_df = pd.read_csv('user_submissions.csv')
    # Filter out placeholder records (where charges are 0) or update them if you have actual charges
    user_df = user_df[user_df['charges'] > 0] 
    df = pd.concat([df, user_df], ignore_index=True)



# 3. Handle Categorical Features using One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Separate features (all columns except charges) and target
X = df_encoded.drop(columns=['charges'])
y = df_encoded['charges']

# Save the exact column structure so Flask can match it later
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

# 4. Create and Train the XGBoost Regressor Model
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)
model.fit(X, y)

# 5. Save the trained model
joblib.dump(model, 'model.pkl')

print("XGBoost model successfully trained with all features and saved!")