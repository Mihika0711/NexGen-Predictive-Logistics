# 🚚 NexGen – Predictive Logistics Dashboard

> An **AI-powered logistics delay prediction system** designed to optimize supply chain efficiency by forecasting delays in deliveries using **XGBoost** and **Streamlit**.

---

## 🧩 Overview
NexGen Predictive Logistics leverages machine learning to predict potential delivery delays based on real-time operational parameters such as cost, distance, traffic, and weather.  
This helps logistics companies **proactively mitigate risks**, **reduce operational costs**, and **enhance customer satisfaction**.

---

## ⚙️ Tech Stack
- **Python 3.10+**
- **Streamlit** – Interactive dashboard
- **XGBoost** – Predictive model
- **Pandas, NumPy** – Data preprocessing
- **Joblib** – Model serialization

---

## 📁 Project Structure
Case study internship data/
├── app.py # Streamlit dashboard
├── cost_breakdown.csv # Fuel, labor, and maintenance cost data
├── customer_feedback.csv # Delivery feedback and satisfaction
├── delivery_performance.csv# Historical delivery records
├── orders.csv # Order details dataset
├── routes_distance.csv # City route distances
├── vehicle_fleet.csv # Fleet maintenance data
├── warehouse_inventory.csv # Warehouse stock records
├── requirements.txt # Project dependencies

models/
├── xgb_delay.pkl # Trained XGBoost model 
└── feature_cols.pkl # Model feature mapping

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

git clone https://github.com/Mihika0711/NexGen-Predictive-Logistics.git
### 2️⃣ Navigate to the Folder
cd "NexGen-Predictive-Logistics/Case study internship data"
### 3️⃣ Install Dependencies
pip install -r requirements.txt
### 4️⃣ Run the Dashboard
streamlit run app.py
## 📊 Features

✅ Predict delivery delays based on cost, distance, and external factors
✅ Interactive Streamlit interface with input sliders and dropdowns
✅ Instant visualization of delay predictions
✅ Modular model design for easy retraining with new data

## 🧠 Model Information

Algorithm: XGBoost Classifier

Input Features: Fuel cost, labor, vehicle maintenance, distance, traffic delay, weather impact, etc.

Output: is_delayed → 0 (On-time) / 1 (Delayed)

Accuracy: ~94% on validation data

Note: Model files (xgb_delay.pkl, feature_cols.pkl) are excluded from GitHub due to size limits but are required to run predictions locally.

## 🧾 Example Prediction
Feature	Example Value
Fuel Cost (INR)	200
Labor Cost (INR)	150
Vehicle Maintenance (INR)	400
Distance (KM)	500
Traffic Delay (Minutes)	30
Weather Impact	Moderate

Predicted Output: ✅ On-Time Delivery
## 📸 Dashboard Preview

Developed using Streamlit with a responsive dark-themed UI.
## 👩‍💻 Author

Mihika Arora
🎓 Final Year B.Tech, Computer & Communication Engineering
🏫 Manipal University Jaipur (2022–2026)
💡 AI/ML | Deep Learning | Data Analytics

