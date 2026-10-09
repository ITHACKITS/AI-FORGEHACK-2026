import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

print("🌍 [1] Membaca Data TITIK API ASLI (NASA FIRMS - GLOBAL)...")
# Membaca data satelit MODIS asli 7 Hari Terakhir dari seluruh dunia
data_path = "data/nasa_global_fires.csv"
if not os.path.exists(data_path):
    print("❌ File data/nasa_global_fires.csv tidak ditemukan!")
    print("Tunggu script mendownload atau pastikan file sudah ada.")
    exit()

df_fires = pd.read_csv(data_path)

# Menggunakan kolom latitude, longitude, dan FRP (Fire Radiative Power)
# Data dari NASA ini semuanya adalah titik api positif (Kebakaran = 1)
df_fires = df_fires[['latitude', 'longitude', 'frp', 'confidence']].dropna()
df_fires['fire_occurred'] = 1

print(f"🔥 Ditemukan {len(df_fires)} titik api AKTIF di seluruh dunia dalam 7 hari terakhir!")

print("\n🌿 [2] Men-generate Data 'Aman' (Negative Samples) untuk Pelatihan...")
# Machine Learning butuh belajar dari daerah yang AMAN juga, bukan cuma yang kebakaran.
# Kita buat titik acak di seluruh dunia (Safe Zones)
n_safe = len(df_fires) * 2  # Bikin data aman 2x lipat lebih banyak
safe_lat = np.random.uniform(-90.0, 90.0, n_safe)
safe_lon = np.random.uniform(-180.0, 180.0, n_safe)

df_safe = pd.DataFrame({
    'latitude': safe_lat,
    'longitude': safe_lon,
    'frp': 0.0,
    'confidence': 100, # Yakin 100% aman
    'fire_occurred': 0
})

# Gabungkan data Kebakaran (Satelit Asli) dengan data Aman
df_all = pd.concat([df_fires, df_safe], ignore_index=True)

# Acak urutan datanya
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total Dataset Latih Global: {len(df_all)} baris.")

print("\n🧠 [3] Melatih Model Machine Learning AI (Random Forest)...")
# Fitur yang dipakai: Latitude & Longitude
X = df_all[['latitude', 'longitude']]
y = df_all['fire_occurred']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n📈 [4] Hasil Evaluasi Model AI (Data Global Asli):")
print(f"Akurasi: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# Menyimpan Model
model_dir = "app/models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "fire_predictor_global.pkl")

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"💾 Model berhasil disimpan di: {model_path}")
print("🚀 Selesai! Model ini sudah mempelajari Peta Panas GLOBAL (Bukan cuma Indonesia) dari satelit NASA asli!")
