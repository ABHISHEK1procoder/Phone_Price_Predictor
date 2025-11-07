📱 Mobile Price Prediction using Machine Learning
This project predicts the price range of mobile phones based on their specifications using a Machine Learning model built with Logistic Regression.
It is deployed using Streamlit for an interactive and user-friendly web interface.

🚀 Overview
The goal of this project is to help users estimate the price category of a mobile phone by analyzing its features such as battery capacity, RAM, camera quality, screen size, and connectivity options.

The model classifies phones into four categories:

Code	Price Category
0	Low Cost 💰
1	Medium Cost 💵
2	High Cost 💸
3	Very High Cost 💎
🧠 Model Details
Algorithm Used: Logistic Regression
Libraries: scikit-learn, pandas, numpy, streamlit, joblib
Data Split: 80% training, 20% testing
Scaler Used: StandardScaler (saved as scalar.joblib)
Model File: mobile_price_model.pkl
Accuracy: 96.5% on test data
⚙️ Features Used for Prediction
Feature	Description
battery_power	Total energy a battery can store (in mAh)
blue	Bluetooth support (1 = Yes, 0 = No)
clock_speed	Processor speed (in GHz)
dual_sim	Dual SIM support (1 = Yes, 0 = No)
fc	Front camera megapixels
four_g	4G support (1 = Yes, 0 = No)
int_memory	Internal memory (in GB)
m_dep	Mobile depth (in cm)
mobile_wt	Mobile weight (in grams)
n_cores	Number of processor cores
pc	Primary camera megapixels
px_height	Pixel resolution height
px_width	Pixel resolution width
ram	RAM (in MB)
sc_h	Screen height
sc_w	Screen width
talk_time	Longest time the phone can be used after a full charge
three_g	3G support (1 = Yes, 0 = No)
touch_screen	Touchscreen feature (1 = Yes, 0 = No)
wifi	Wi-Fi support (1 = Yes, 0 = No)
🧩 How It Works
The model is trained on mobile specifications using Logistic Regression.
The input data is scaled using StandardScaler to maintain consistency.
Users enter mobile specifications in the Streamlit interface.
The model predicts the price category of the phone in real-time.
