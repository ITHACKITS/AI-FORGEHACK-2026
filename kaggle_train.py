# ============================================================
# NASA FIRMS GLOBAL FIRE — ADVANCED AI TRAINING PIPELINE
# ForgeHack 2026 | ITHACKITS | UdaraSiaga
# Pipeline: ZIP->CSV -> Feature Eng -> CrossVal -> 3 Models -> Best Model
# ============================================================

import glob, os, pickle, warnings, zipfile
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

print("="*65)
print("NASA FIRMS GLOBAL FIRE AI - Advanced Training Pipeline")
print("ForgeHack 2026 | UdaraSiaga | ITHACKITS")
print("="*65)

# [1] Deteksi environment
KAGGLE_INPUT = "/kaggle/input"
if os.path.exists(KAGGLE_INPUT):
    BASE_PATH, OUTPUT_PATH = KAGGLE_INPUT, "/kaggle/working"
    print("Environment: KAGGLE CLOUD")
else:
    BASE_PATH, OUTPUT_PATH = "data", "."
    print("Environment: LOKAL")

# [2] Baca data NASA FIRMS
COLUMNS_NEEDED = ["latitude","longitude","brightness","frp","confidence","daynight"]
SAMPLE_PER_FILE = 200_000
CONFIDENCE_MAP = {"l":33,"low":33,"n":66,"nominal":66,"h":90,"high":90}

print(f"\n[1] Memindai dataset dari {BASE_PATH} ...")
zip_files = sorted(glob.glob(f"{BASE_PATH}/**/*.zip", recursive=True)) or sorted(glob.glob(f"{BASE_PATH}/*.zip"))
csv_files = sorted(glob.glob(f"{BASE_PATH}/**/*.csv", recursive=True)) or sorted(glob.glob(f"{BASE_PATH}/*.csv"))
print(f"    ZIP: {len(zip_files)} | CSV: {len(csv_files)}")

def read_fire_csv(fp, name=""):
    try:
        header = pd.read_csv(fp, nrows=0)
        cols = [c for c in COLUMNS_NEEDED if c in header.columns]
        if "latitude" not in cols or "longitude" not in cols:
            return None
        df = pd.read_csv(fp, usecols=cols, low_memory=False)
        df = df.dropna(subset=["latitude","longitude"])
        for col, val in [("frp",15.0),("brightness",320.0),("confidence",50),("daynight","D")]:
            if col not in df.columns:
                df[col] = val
        df["confidence"] = df["confidence"].apply(
            lambda x: CONFIDENCE_MAP.get(str(x).strip().lower(), x)
            if not str(x).strip().replace(".","").isdigit() else x)
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(50)
        df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0)
        df["brightness"] = pd.to_numeric(df["brightness"], errors="coerce").fillna(320)
        df = df[df["latitude"].between(-90,90) & df["longitude"].between(-180,180)]
        df = df[df["frp"].between(0,5000) & df["brightness"].between(250,500)]
        if len(df) > SAMPLE_PER_FILE:
            df = df.sample(n=SAMPLE_PER_FILE, random_state=42)
        return df
    except Exception as e:
        print(f"    GAGAL: {name} -> {e}")
        return None

all_dfs = []
for zpath in zip_files:
    print(f"\n  ZIP: {os.path.basename(zpath)}")
    try:
        with zipfile.ZipFile(zpath) as zf:
            for inner in [f for f in zf.namelist() if f.lower().endswith(".csv")]:
                print(f"    Membaca: {inner} ...", end=" ")
                with zf.open(inner) as f:
                    df = read_fire_csv(f, inner)
                    if df is not None:
                        all_dfs.append(df)
                        print(f"OK {len(df):,} baris")
    except zipfile.BadZipFile:
        print("    SKIP: file ZIP rusak")

for cpath in csv_files:
    name = os.path.basename(cpath)
    if "nasa_global_fires" in name:
        continue
    print(f"\n  CSV: {name} ...", end=" ")
    df = read_fire_csv(cpath, name)
    if df is not None:
        all_dfs.append(df)
        print(f"OK {len(df):,} baris")

if not all_dfs:
    raise FileNotFoundError("TIDAK ADA DATA! Pastikan dataset NASA FIRMS sudah di-add ke Notebook.")

df_fires = pd.concat(all_dfs, ignore_index=True)
df_fires["fire_occurred"] = 1
print(f"\nTotal titik api (2020-2025): {len(df_fires):,}")

# [3] Feature Engineering Lanjutan
print("\n[2] Feature Engineering (12 fitur) ...")
def engineer_features(df):
    df = df.copy()
    df["is_day"] = df.get("daynight", pd.Series("D", index=df.index)).astype(str).str.strip().str.upper().eq("D").astype(int)
    df["dist_equator"] = df["latitude"].abs()
    df["is_northern"] = (df["latitude"] >= 0).astype(int)
    df["is_east_hemisphere"] = (df["longitude"] >= 0).astype(int)
    df["brightness_delta"] = (df["brightness"] - 330.0).clip(lower=0)
    df["fire_intensity"] = df["frp"] * (df["confidence"] / 100.0)
    df["frp_log"] = np.log1p(df["frp"])
    return df

df_fires = engineer_features(df_fires)
FEATURES = ["latitude","longitude","brightness","frp","frp_log","confidence","is_day",
            "dist_equator","is_northern","is_east_hemisphere","brightness_delta","fire_intensity"]
print(f"    Fitur: {FEATURES}")

# [4] Negative Samples (Zona Aman)
print(f"\n[3] Membuat {len(df_fires):,} Negative Samples ...")
n_safe = len(df_fires)
rng = np.random.default_rng(42)
lat_s, lon_s = rng.uniform(-90,90,n_safe), rng.uniform(-180,180,n_safe)
frp_s = np.zeros(n_safe)
bright_s = rng.uniform(270,305,n_safe)
conf_s = rng.uniform(0,30,n_safe)
df_safe = pd.DataFrame({
    "latitude":lat_s,"longitude":lon_s,"brightness":bright_s,
    "frp":frp_s,"confidence":conf_s,"fire_occurred":0,
    "is_day":rng.integers(0,2,n_safe),
    "dist_equator":np.abs(lat_s),
    "is_northern":(lat_s>=0).astype(int),
    "is_east_hemisphere":(lon_s>=0).astype(int),
    "brightness_delta":np.clip(bright_s-330,0,None),
    "fire_intensity":frp_s*conf_s/100.0,
    "frp_log":np.log1p(frp_s),
})
df_all = pd.concat([df_fires,df_safe],ignore_index=True).sample(frac=1,random_state=42).reset_index(drop=True)
df_all = df_all.dropna(subset=FEATURES)
print(f"    Total dataset: {len(df_all):,} baris (balanced 50/50)")

# [5] Split & Scale
X = df_all[FEATURES].values
y = df_all["fire_occurred"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n[4] Split: Train={len(X_train):,} | Test={len(X_test):,}")

# [6] Training 3 Model
print("\n[5] Training 3 Model ...")
print("    Perkiraan waktu: 10-25 menit di Kaggle")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model 1: Random Forest
print("\n    [1/3] Random Forest ...")
rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=4,
                             min_samples_leaf=2, max_features="sqrt",
                             n_jobs=-1, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
rf_cv = cross_val_score(rf, X_scaled, y, cv=cv, scoring="f1", n_jobs=-1)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print(f"    OK Acc={rf_acc*100:.2f}%  CV-F1={rf_cv.mean():.4f}  AUC={rf_auc:.4f}")

# Model 2: Gradient Boosting
print("\n    [2/3] Gradient Boosting ...")
gbm = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  subsample=0.8, min_samples_split=4, random_state=42)
gbm.fit(X_train, y_train)
gbm_cv = cross_val_score(gbm, X_scaled, y, cv=cv, scoring="f1", n_jobs=-1)
gbm_acc = accuracy_score(y_test, gbm.predict(X_test))
gbm_auc = roc_auc_score(y_test, gbm.predict_proba(X_test)[:,1])
print(f"    OK Acc={gbm_acc*100:.2f}%  CV-F1={gbm_cv.mean():.4f}  AUC={gbm_auc:.4f}")

# Model 3: Voting Ensemble
print("\n    [3/3] Voting Ensemble (RF + GBM) ...")
ensemble = VotingClassifier(estimators=[("rf",rf),("gbm",gbm)], voting="soft", weights=[1.5,1.0], n_jobs=-1)
ensemble.fit(X_train, y_train)
ens_cv = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring="f1", n_jobs=-1)
ens_acc = accuracy_score(y_test, ensemble.predict(X_test))
ens_auc = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:,1])
print(f"    OK Acc={ens_acc*100:.2f}%  CV-F1={ens_cv.mean():.4f}  AUC={ens_auc:.4f}")

# [7] Pilih model terbaik
candidates = {
    "RandomForest":     (rf,       rf_acc,  rf_cv.mean(),  rf_auc),
    "GradientBoosting": (gbm,      gbm_acc, gbm_cv.mean(), gbm_auc),
    "VotingEnsemble":   (ensemble, ens_acc, ens_cv.mean(), ens_auc),
}
best_name = max(candidates, key=lambda k: 0.4*candidates[k][1] + 0.4*candidates[k][2] + 0.2*candidates[k][3])
best_model, best_acc, best_cv, best_auc = candidates[best_name]
print(f"\n[6] Model Terbaik: {best_name}")
print(f"    Akurasi={best_acc*100:.2f}%  CV-F1={best_cv:.4f}  AUC={best_auc:.4f}")

# [8] Evaluasi Detail
print(f"\n[7] Evaluasi Detail '{best_name}':")
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]
print("="*55)
print(f"  AKURASI : {accuracy_score(y_test,y_pred)*100:.2f}%")
print(f"  ROC-AUC : {roc_auc_score(y_test,y_prob):.4f}")
print("="*55)
print(classification_report(y_test, y_pred, target_names=["AMAN","KEBAKARAN"]))
cm = confusion_matrix(y_test, y_pred)
tn,fp,fn,tp = cm.ravel()
print(f"  Precision: {tp/(tp+fp)*100:.2f}%")
print(f"  Recall   : {tp/(tp+fn)*100:.2f}%  <- Harus tinggi!")

print("\n  Feature Importance (Random Forest):")
for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:<20}: {'|'*int(imp*50)} ({imp:.4f})")

# [9] Simpan model bundle
bundle = {
    "model":rf if best_name=="RandomForest" else (gbm if best_name=="GradientBoosting" else ensemble),
    "scaler":scaler, "features":FEATURES, "accuracy":best_acc,
    "roc_auc":best_auc, "cv_f1":best_cv, "model_type":best_name,
    "training_samples":len(X_train),
    "data_source":"NASA FIRMS VIIRS S-NPP C2 (2020-2025) Global",
    "class_labels":{0:"AMAN",1:"KEBAKARAN"},
}
bundle_path = os.path.join(OUTPUT_PATH, "fire_predictor_global.pkl")
with open(bundle_path,"wb") as f:
    pickle.dump(bundle, f)
size_mb = os.path.getsize(bundle_path)/(1024*1024)
print(f"\n[8] Model disimpan: {bundle_path} ({size_mb:.1f} MB)")

print("\n"+"="*65)
print("TRAINING SELESAI!")
print(f"  Model  : {best_name}")
print(f"  Akurasi: {best_acc*100:.2f}%  |  AUC: {best_auc:.4f}")
print(f"  Output : fire_predictor_global.pkl ({size_mb:.1f} MB)")
print("\nLangkah selanjutnya:")
print("  1. Klik tab Output di panel kanan Kaggle")
print("  2. Download fire_predictor_global.pkl")
print("  3. Taruh di folder proyek lokal")
print("="*65)
