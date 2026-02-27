import os
import joblib
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= LOAD MODELS =================
model1 = joblib.load(os.path.join(BASE_DIR, "models/model1_fertilizer_classifier.pkl"))
model2 = joblib.load(os.path.join(BASE_DIR, "models/model2_fertilizer_quantity.pkl"))

feature_columns = joblib.load(os.path.join(BASE_DIR, "models/feature_columns.pkl"))
categorical_cols = joblib.load(os.path.join(BASE_DIR, "models/categorical_cols.pkl"))
numeric_cols = joblib.load(os.path.join(BASE_DIR, "models/numeric_cols.pkl"))

y_encoder_m1 = joblib.load(os.path.join(BASE_DIR, "models/y_label_encoder_m1.pkl"))

# ================= DOMAIN DATA =================
FERTILIZER_CONTENT = {
    "Urea": (46, 0, 0),
    "DAP": (18, 46, 0),
    "SSP": (0, 16, 0),
    "MOP": (0, 0, 60),
    "White Potash": (0, 0, 60),
    "Ammonium Sulphate": (21, 0, 0),

    "10:10:10 NPK": (10, 10, 10),
    "10:26:26 NPK": (10, 26, 26),
    "12:32:16 NPK": (12, 32, 16),
    "13:32:26 NPK": (13, 32, 26),
    "18:46:00 NPK": (18, 46, 0),
    "19:19:19 NPK": (19, 19, 19),
    "20:20:20 NPK": (20, 20, 20),
    "50:26:26 NPK": (50, 26, 26),

    "Sulphur": (0, 0, 0),
    "Magnesium Sulphate": (0, 0, 0),
    "Ferrous Sulphate": (0, 0, 0),
    "Chilated Micronutrient": (0, 0, 0),
    "Hydrated Lime": (0, 0, 0),
}

FERTILIZER_COST = {
    "Urea": 6,
    "DAP": 27,
    "SSP": 5,
    "MOP": 17,
    "White Potash": 17,
    "Ammonium Sulphate": 9,

    "10:10:10 NPK": 18,
    "10:26:26 NPK": 24,
    "12:32:16 NPK": 26,
    "13:32:26 NPK": 28,
    "18:46:00 NPK": 27,
    "19:19:19 NPK": 30,
    "20:20:20 NPK": 32,
    "50:26:26 NPK": 35,

    "Sulphur": 8,
    "Magnesium Sulphate": 20,
    "Ferrous Sulphate": 15,
    "Chilated Micronutrient": 120,
    "Hydrated Lime": 3,
}

IDEAL_NPK = {
    "Cotton": {"N": 120, "P": 60, "K": 60},
    "Ginger": {"N": 75, "P": 50, "K": 50},
    "Gram": {"N": 20, "P": 40, "K": 20},
    "Grapes": {"N": 100, "P": 50, "K": 100},
    "Groundnut": {"N": 25, "P": 50, "K": 25},
    "Jowar": {"N": 80, "P": 40, "K": 40},
    "Maize": {"N": 120, "P": 60, "K": 40},
    "Masoor": {"N": 20, "P": 40, "K": 20},
    "Moong": {"N": 20, "P": 40, "K": 20},
    "Rice": {"N": 100, "P": 50, "K": 50},
    "Soybean": {"N": 30, "P": 60, "K": 40},
    "Sugarcane": {"N": 150, "P": 60, "K": 80},
    "Tur": {"N": 25, "P": 50, "K": 25},
    "Turmeric": {"N": 60, "P": 50, "K": 50},
    "Urad": {"N": 20, "P": 40, "K": 20},
    "Wheat": {"N": 120, "P": 60, "K": 40},
}

# ================= HELPERS =================
def compute_deficiency(crop, N, P, K):
    ideal = IDEAL_NPK.get(crop, {"N": 100, "P": 50, "K": 50})
    return {
        "N": max(0, ideal["N"] - N),
        "P": max(0, ideal["P"] - P),
        "K": max(0, ideal["K"] - K),
    }

def soil_health_warnings(defs):
    if defs["N"] == defs["P"] == defs["K"] == 0:
        return "Soil nutrients are balanced. No fertilizer required."
    if defs["N"] > 60:
        return "Severe Nitrogen deficiency detected."
    if defs["P"] > 30:
        return "Low Phosphorus – poor root development."
    if defs["K"] > 40:
        return "Low Potassium – increased disease risk."
    return "Moderate nutrient deficiency detected."

def nutrient_supply(fertilizer, qty):
    n, p, k = FERTILIZER_CONTENT.get(fertilizer, (0, 0, 0))
    return {
        "N": round(qty * n / 100, 2),
        "P": round(qty * p / 100, 2),
        "K": round(qty * k / 100, 2),
    }

# ================= MAIN ENGINE =================
def recommend_fertilizer(data):

    data = {
        "soil_color": str(data["soil_color"]),
        "crop": str(data["crop"]),
        "nitrogen": float(data["nitrogen"]),
        "phosphorus": float(data["phosphorus"]),
        "potassium": float(data["potassium"]),
    }

    # ---------- MODEL 1 ----------
    X1 = pd.DataFrame([{
        "Soil_color": data["soil_color"],
        "Nitrogen": data["nitrogen"],
        "Phosphorus": data["phosphorus"],
        "Potassium": data["potassium"],
        "Crop": data["crop"]
    }])[feature_columns]

    probs = model1.predict_proba(X1)[0]
    labels = y_encoder_m1.inverse_transform(model1.classes_.astype(int))

    top_idx = np.argsort(probs)[::-1][:2]
    top_ferts = [(labels[i], float(probs[i])) for i in top_idx]

    defs = compute_deficiency(
        data["crop"],
        data["nitrogen"],
        data["phosphorus"],
        data["potassium"]
    )

    results = []

    for fert, conf in top_ferts:

        if fert not in FERTILIZER_CONTENT:
            continue

        X2 = pd.DataFrame([{
            "Crop": data["crop"],
            "Fertilizer": fert,
            "N_def": defs["N"],
            "P_def": defs["P"],
            "K_def": defs["K"]
        }])

        qty = float(model2.predict(X2)[0])
        qty = round(max(qty, 0), 2)

        results.append({
            "fertilizer": fert,
            "confidence": round(conf, 3),
            "quantity": qty,
            "cost_per_kg": FERTILIZER_COST.get(fert, 0),
            "total_cost": round(qty * FERTILIZER_COST.get(fert, 0), 2),
            "nutrients": nutrient_supply(fert, qty)
        })

    if not results:
        return {"error": "No fertilizer recommendation available"}

    best = results[0]

        # 🌱 Balanced soil case
    if defs["N"] == 0 and defs["P"] == 0 and defs["K"] == 0:
        return {
            "fertilizer": "None",
            "confidence": round(max(p for _, p in top_ferts), 3),
            "quantity": 0,
            "cost_per_kg": 0,
            "total_cost": 0,
            "nutrients": {"N": 0, "P": 0, "K": 0},
            "deficiency": defs,
            "health_note": "Soil nutrients are balanced. No fertilizer application required.",
            "alternatives": [],
            "balanced": True
     }

    return {
        "fertilizer": best["fertilizer"],
        "confidence": best["confidence"],
        "quantity": best["quantity"],
        "cost_per_kg": best["cost_per_kg"],
        "total_cost": best["total_cost"],
        "nutrients": best["nutrients"],
        "deficiency": defs,
        "health_note": soil_health_warnings(defs),
        "alternatives": results
    }
