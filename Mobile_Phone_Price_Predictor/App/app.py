import streamlit as st
import numpy as np
import joblib

# ==============================
# Load model and scaler
# ==============================
model = joblib.load("model.joblib")    # your trained model
scaler = joblib.load("scaler.joblib")            # your saved scaler

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
