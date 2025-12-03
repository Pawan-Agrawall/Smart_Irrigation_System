import pickle
import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai
import requests
from datetime import datetime

st.set_page_config(
    page_title="Smart Agriculture Irrigation System",
    page_icon="💧",
    layout="wide",
)


def get_live_sensor_data():
    try:
        r = requests.get("http://localhost:5000/latest", timeout=2)
        return r.json()
    except:
        return None


class AgricultureGeminiChatbot:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

        self.system_prompt = """
        You are an agriculture expert specializing in irrigation, soil, crops, and sustainable farming.
        Provide practical, region-agnostic irrigation advice.
        """

    def get_response(self, question, context=None):
        try:
            context_info = ""
            if context:
                context_info = "\nFIELD CONTEXT:\n" + "\n".join([f"- {k}: {v}" for k, v in context.items()])
            prompt = f"{self.system_prompt}\n{context_info}\n\nUser Question: {question}\nAnswer:"
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ Error: {e}"


@st.cache_resource
def load_model():
    try:
        with open("rf_irrigation_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


def predict_irrigation(model, input_data):
    try:
        return model.predict(input_data)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def irrigation_level(pred):
    if pred < 50000:
        return "Low", "🟢", "Minimal irrigation needed", "low"
    elif pred < 120000:
        return "Moderate", "🟡", "Standard irrigation recommended", "medium"
    else:
        return "High", "🔴", "Substantial irrigation required", "high"


def main():
    st.markdown('<h1 style="text-align:center;">🌾 Smart Agriculture Irrigation System 💧</h1>', unsafe_allow_html=True)

    api_key = st.sidebar.text_input("🔑 Gemini API Key:", type="password")
    model = load_model()

    tab1, tab2 = st.tabs(["💧 Live Irrigation Prediction", "🤖 AI Agriculture Advisor"])


    with tab1:
        st.markdown("### 🌿 Live Field Data From ESP32")

        live = get_live_sensor_data()

        if live and live["temperature"] is not None:
            st.success("🔥 Live sensor data received from ESP32!")

            temperature = live["temperature"]
            humidity = live["humidity"]
            soil_moisture = live["soil"]

            col1, col2, col3 = st.columns(3)
            col1.metric("🌡 Temperature (°C)", temperature)
            col2.metric("💧 Humidity (%)", humidity)
            col3.metric("🌱 Soil Moisture (%)", soil_moisture)

            soil_type = st.selectbox("Soil Type", ["Clay","Loam","Sandy","Silt"])
            crop_type = st.selectbox("Crop Type", ["Rice","Wheat","Maize","Sugarcane","Cotton","Millet","Pulses","Soybean","Groundnut"])

            input_df = pd.DataFrame({
                "Soil_Type": [soil_type],
                "Soil_Moisture": [soil_moisture],
                "Temperature": [temperature],
                "Humidity": [humidity],
                "Crop_Type": [crop_type],
            })

            if model:
                pred = predict_irrigation(model, input_df)
                level, emoji, advice, css_class = irrigation_level(pred)

                st.markdown("### 💧 Prediction Based on LIVE Data")
                st.markdown(f"**{emoji} Irrigation Level:** {level}")
                st.markdown(f"**💦 Water Needed:** {pred:,.2f} litres/acre")
                st.markdown(f"**🧭 Recommendation:** {advice}")

                st.session_state.context = {
                    "Soil Type": soil_type,
                    "Soil Moisture": f"{soil_moisture}%",
                    "Temperature": f"{temperature}°C",
                    "Humidity": f"{humidity}%",
                    "Crop Type": crop_type,
                    "Predicted Water Need": f"{pred:,.2f} L/acre",
                    "Irrigation Level": level
                }
        else:
            st.warning("⏳ Waiting for ESP32 sensor data...")


    with tab2:
        st.markdown("### 🤖 Ask the AI Agriculture Expert")

        if api_key:
            chatbot = AgricultureGeminiChatbot(api_key)
            question = st.text_area("Ask your question:")

            if st.button("💬 Get Expert Advice"):
                with st.spinner("Generating response..."):
                    context = st.session_state.get("context", {})
                    st.write(chatbot.get_response(question, context))
        else:
            st.info("Enter Gemini API Key in sidebar.")

if __name__ == "__main__":
    main()
