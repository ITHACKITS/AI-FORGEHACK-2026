"""
test_fire_ai.py

Script pengujian inferensi AI Fire Classifier (NASA FIRMS 2020-2025).
"""

from app.engine.fire_classifier import FireClassifier

def main():
    print("=" * 60)
    print("🔥 UJI COBA INFERENSI MODEL AI FIRE CLASSIFIER (UDARASIAGA)")
    print("=" * 60)

    clf = FireClassifier()
    print("\n[1] Metadata Model:")
    for k, v in clf.metadata.items():
        if k != "features":
            print(f"    - {k}: {v}")

    print("\n[2] Uji Titik 1: Titik Api Nyata (Kalimantan Tengah)")
    res1 = clf.predict_point(
        latitude=-1.85,
        longitude=113.75,
        brightness=348.5,
        frp=52.3,
        confidence=95.0,
        is_day=True
    )
    print(f"    Status      : {'🔥 ' + ('KEBAKARAN' if res1['is_fire'] else 'AMAN')}")
    print(f"    Probabilitas: {res1['fire_probability']*100:.2f}%")
    print(f"    Tingkat Risk: {res1['risk_level']}")
    print(f"    Model Engine: {res1['model_engine']}")

    print("\n[3] Uji Titik 2: Titik Aman / Dingin (Bukan Titik Api)")
    res2 = clf.predict_point(
        latitude=-1.85,
        longitude=113.75,
        brightness=295.0,
        frp=0.0,
        confidence=10.0,
        is_day=True
    )
    print(f"    Status      : {'🌿 ' + ('KEBAKARAN' if res2['is_fire'] else 'AMAN')}")
    print(f"    Probabilitas: {res2['fire_probability']*100:.2f}%")
    print(f"    Tingkat Risk: {res2['risk_level']}")
    print(f"    Model Engine: {res2['model_engine']}")

    print("\n" + "=" * 60)
    print("✅ PENGUJIAN SELESAI & SUKSES!")
    print("=" * 60)

if __name__ == "__main__":
    main()
