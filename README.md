
# 🌱 Smart Irrigation System (ESP32 + Machine Learning + Flask + Streamlit + AI Advisor)
# 🎥 Project Demo (Video)
https://github.com/user-attachments/assets/9d3460dc-c2ee-4f8d-a5f3-53374fb0acf6


This project is a complete **IoT + ML Smart Irrigation System** integrating:

- 🌡️ **ESP32 + DHT22 + Soil Moisture Sensor**
- 🌐 **Flask API Backend**
- 🤖 **Random Forest ML Model** (Irrigation Prediction)
- 🖥️ **Streamlit Dashboard**
- 🧠 **Gemini-powered AI Agriculture Advisor**
- 📡 **Real-time communication between hardware & dashboard**

---

# 📸 Hardware Setup

<img src="https://github.com/Pawan-Agrawall/Smart_Irrigation_System/blob/main/285bb8e4-0681-4aab-bef9-68c6c21f9f15.jpg?raw=true" width="600" style="transform: rotate(-90deg);" />

**Components Used:**
- ESP32 Dev Board  
- DHT22 Temperature & Humidity Sensor  
- Soil Moisture Analog Sensor  
- Jumper Wires  
- Breadboard  

---


# 📁 Project Structure

├── demo.py # Flask backend API
├── test.ino # ESP32 code (DHT22 + Soil Sensor)
├── train.py # ML model training script
├── merged_irrigation_dataset_5000.csv
├── rf_irrigation_model.pkl # Trained Random Forest model
├── app.py # Streamlit dashboard + Gemini AI
├── images/
│ ├── hardware.jpg
│ ├── screenshot1.png
│ ├── screenshot2.png
└── README.md

---

# 🧠 Gemini AI Advisor (IMPORTANT)

To use the **AI Agriculture Chatbot**, you must enter a:

👉 **Google Gemini API Key**

In the Streamlit app sidebar.

Without this, the chatbot will NOT respond.

---

# 🚀 How to Run The Project

## 1️⃣ Install dependencies
```
pip install flask streamlit scikit-learn pandas numpy requests
```

## 2️⃣ Train the ML Model
```
python train.py
```

This generates:
rf_irrigation_model.pkl


---

## 3️⃣ Start the Flask Backend
```
python demo.py
```


Server default:
```
http://localhost:5000
```

---

## 4️⃣ Upload code to ESP32

Open `test.ino` in Arduino IDE and update:

- WiFi SSID  
- WiFi Password  
- Flask Server IP  

Then upload to ESP32.

ESP32 sends JSON like:

```json
{
  "temperature": 29.4,
  "humidity": 61,
  "soil": 387
}
```
5️⃣ Run the Streamlit Dashboard
streamlit run app.py
Opens at:
```
http://localhost:8501
```
Dashboard shows:

Live sensor data

ML-based irrigation recommendation

Water needed

Crop & soil selector

Gemini AI Advisor

🌐 Flask API Endpoint
POST /data
```
{
  "temperature": 22.7,
  "humidity": 58.4,
  "soil": 76
}
```
Response
```
{
  "prediction": "Moderate Irrigation Required",
  "water_needed": 74129.17,
  "level": 2
}
```
🖥️ Streamlit UI Preview
<img src="https://github.com/Pawan-Agrawall/Smart_Irrigation_System/blob/main/Screenshot%202025-11-29%20130759.png?raw=true" width="600" /> <br><img src="https://github.com/Pawan-Agrawall/Smart_Irrigation_System/blob/main/Screenshot%202025-11-29%20130714.png?raw=true" width="600" /><br>
💡 Features
🌡️ Real-time sensor data from ESP32

🤖 ML-based irrigation prediction

📊 Beautiful Streamlit dashboard

🧠 Gemini-powered Agriculture Advisor

🌾 Crop & Soil Selection system

🔧 Fully automatic irrigation recommendation

🎥 Video demo included

📡 Seamless hardware-to-cloud system

🔧 Troubleshooting
❗ ESP32 not connecting
Check WiFi name & password

Correct COM port

ESP32 Dev Module selected

❗ Streamlit not updating
Flask IP mismatch

ESP32 not posting data

❗ AI chatbot not working
Enter your Gemini API Key.

🤝 Contributing
Pull requests and suggestions are welcome.

📜 License
MIT License.

🎉 Thank You!
