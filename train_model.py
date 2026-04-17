import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── 1. Load Data ──────────────────────────────────────────────────────
df = pd.read_csv("high-speed trains operation data (1).csv")

# ── 2. Clean ──────────────────────────────────────────────────────────
df['stop_time'] = df['stop_time'].replace('----', 0)
df['arrival_delay']   = df['arrival_delay'].abs()
df['departure_delay'] = df['departure_delay'].abs()
df['final_delay']     = df['arrival_delay'] + df['departure_delay'] + df['stop_time'].astype(float)

df['date']  = pd.to_datetime(df['date'])
df['year']  = df['date'].dt.year
df['month'] = df['date'].dt.month

def time_to_minutes(t):
    try:
        return t.hour * 60 + t.minute + t.second / 60
    except:
        return 0

for col in ['scheduled_arrival_time', 'scheduled_departure_time']:
    df[col + '_minutes'] = df[col].apply(time_to_minutes)

# ── 3. Features ───────────────────────────────────────────────────────
FEATURES = [
    'train_number', 'train_direction', 'station_name', 'wind', 'weather',
    'scheduled_arrival_time_minutes', 'scheduled_departure_time_minutes',
    'month', 'year',
]

df = df.dropna(subset=FEATURES + ['final_delay'])

# ── 4. Encode ─────────────────────────────────────────────────────────
categorical_cols = ['train_number', 'train_direction', 'station_name', 'wind', 'weather']
label_encoders = {}
encoder_classes = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    encoder_classes[col] = list(le.classes_)

# ── 5. Split ──────────────────────────────────────────────────────────
X = df[FEATURES]
y = df['final_delay']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ── 6. Scale ──────────────────────────────────────────────────────────
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 7. Define 2 models ────────────────────────────────────────────────
rf  = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=15, random_state=42, n_jobs=-1)
xgb = XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)

# ── 8. Ensemble ───────────────────────────────────────────────────────
ensemble = VotingRegressor(estimators=[
    ('random_forest', rf),
    ('xgboost', xgb)
])

# ── 9. Train ──────────────────────────────────────────────────────────
print("Training models... please wait")
ensemble.fit(X_train_scaled, y_train)

# ── 10. Evaluate each model ───────────────────────────────────────────
print("\n" + "="*50)
print("Individual Model Results:")
print("="*50)

for name, model in [('Random Forest', rf), ('XGBoost', xgb)]:
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    print(f"{name}:")
    print(f"  MAE : {mean_absolute_error(y_test, pred):.2f} minutes")
    print(f"  R2  : {r2_score(y_test, pred):.4f}")

# ── 11. Evaluate ensemble ─────────────────────────────────────────────
ensemble_pred = ensemble.predict(X_test_scaled)
print("\n" + "="*50)
print("Ensemble (Random Forest + XGBoost):")
print("="*50)
print(f"  MAE : {mean_absolute_error(y_test, ensemble_pred):.2f} minutes")
print(f"  R2  : {r2_score(y_test, ensemble_pred):.4f}")
print("="*50)

# ── 12. Save ──────────────────────────────────────────────────────────
artifacts = {
    'model':           ensemble,
    'scaler':          scaler,
    'label_encoders':  label_encoders,
    'encoder_classes': encoder_classes,
    'features':        FEATURES,
}

with open('model_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\nmodel_artifacts.pkl saved successfully!")
print("Now run: python app_egypt.py")
