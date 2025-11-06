import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import sys

def fatal(msg: str):
    try:
        st.error(msg)
    except Exception:
        print(msg, file=sys.stderr)
    sys.exit(1)

# ==============================
# Load model and scaler (robust)
# ==============================
def find_resource(filename: str) -> Path | None:
    base = Path(__file__).resolve().parent
    for p in (base, *base.parents):
        candidate = p / filename
        if candidate.exists():
            return candidate
    return None

model_path = find_resource("model.joblib")
scaler_path = find_resource("scaler.joblib")

if model_path is None or scaler_path is None:
    missing = [name for name, p in (("model.joblib", model_path), ("scaler.joblib", scaler_path)) if p is None]
    fatal(f"Missing files: {', '.join(missing)}. Place them next to app.py (Mobile_Phone_Price_Predictor/App/) and redeploy.\nRun the app with: streamlit run {Path(__file__).resolve()}")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    fatal(f"Failed to load model/scaler: {e}")

# ==============================
# Streamlit UI Setup
# ==============================
st.set_page_config(page_title="📱 Mobile Price Predictor", layout="centered")
st.title("📱 Mobile Phone Price Predictor")
st.write("Predict the **price range** of a mobile phone based on its specifications.")

st.markdown("""
#### Price Range:
- **Low Cost 💰**
- **Medium Cost 💵**
- **High Cost 💸**
- **Very High Cost 💎**
""")

st.divider()

# ==============================
# Input Fields
# ==============================
col1, col2 = st.columns(2)

with col1:
    battery_power = st.number_input("🔋 Battery Power (mAh)", 500, 20000, 1000)
    blue = st.selectbox("🔵 Bluetooth", [0, 1], format_func=lambda x: "Yes" if x else "No")
    clock_speed = st.number_input("⚙️ Clock Speed (GHz)", 0.1, 3.0, 1.5)
    dual_sim = st.selectbox("📶 Dual SIM", [0, 1], format_func=lambda x: "Yes" if x else "No")
    fc = st.number_input("🤳 Front Camera (MP)", 0, 20, 5)
    four_g = st.selectbox("📡 4G Support", [0, 1], format_func=lambda x: "Yes" if x else "No")
    int_memory = st.number_input("💾 Internal Memory (GB)", 2, 512, 32)
    m_dep = st.number_input("📏 Mobile Depth (cm)", 0.1, 1.0, 0.5)
    mobile_wt = st.number_input("⚖️ Mobile Weight (grams)", 50, 300, 150)
    n_cores = st.number_input("🧮 Number of Cores", 1, 8, 4)

with col2:
    pc = st.number_input("📸 Primary Camera (MP)", 0, 20, 10)
    px_height = st.number_input("🖼️ Pixel Height", 0, 2000, 800)
    px_width = st.number_input("🖼️ Pixel Width", 0, 2000, 1200)
    ram = st.number_input("🧠 RAM (MB)", 256, 8192, 2048)
    sc_h = st.number_input("📱 Screen Height (cm)", 5, 20, 10)
    sc_w = st.number_input("📱 Screen Width (cm)", 1, 20, 5)
    talk_time = st.number_input("⏰ Talk Time (hours)", 2, 50, 10)
    three_g = st.selectbox("📶 3G Support", [0, 1], format_func=lambda x: "Yes" if x else "No")
    touch_screen = st.selectbox("🖐️ Touch Screen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    wifi = st.selectbox("📡 WiFi", [0, 1], format_func=lambda x: "Yes" if x else "No")

# ==============================
# Prepare Input for Prediction
# ==============================
input_data = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory,
                        m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram,
                        sc_h, sc_w, talk_time, three_g, touch_screen, wifi]])

# Scale the input data using the same scaler used in training
scaled_input = scaler.transform(input_data)

# ==============================
# Prediction
# ==============================
if st.button("🔮 Predict Price Range"):
    prediction = model.predict(scaled_input)[0]
    price_labels = {
        0: "Low Cost 💰",
        1: "Medium Cost 💵",
        2: "High Cost 💸",
        3: "Very High Cost 💎"
    }
    st.success(f"Predicted Price Range: **{price_labels[prediction]}**")
