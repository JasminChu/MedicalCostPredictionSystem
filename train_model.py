import kagglehub
import pandas as pd
import xgboost as xgb
import joblib
import os

# 1. Download base dataset from Kaggle
path = kagglehub.dataset_download("mirichoi0218/insurance")
files = os.listdir(path)
data_file = [f for f in files if f.endswith('.csv') or f.endswith('.xlsx')][0]
full_path = os.path.join(path, data_file)

df_base = pd.read_csv(full_path)

# 2. Check for uploaded user submissions and merge if available
upload_path = os.path.join('uploads', 'user_submissions.csv')
if os.path.exists(upload_path):
    df_user = pd.read_csv(upload_path)
    # Keep only records that have actual charges (greater than 0)
    df_user = df_user[df_user['charges'] > 0]
    
    if not df_user.empty:
        print(f"Merging {len(df_user)} user-submitted records into training data.")
        df = pd.concat([df_base, df_user], ignore_index=True)
    else:
        df = df_base
else:
    df = df_base

# 3. Handle Categorical Features using One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

X = df_encoded.drop(columns=['charges'])
y = df_encoded['charges']

# Save column structure for consistency during prediction
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

# 4. Train the XGBoost Regressor Model
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)
model.fit(X, y)

# 5. Save the updated model
joblib.dump(model, 'model.pkl')

print("Model successfully retrained with all data and saved to model.pkl!")