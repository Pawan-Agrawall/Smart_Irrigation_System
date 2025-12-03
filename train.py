import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set page config
st.set_page_config(
    page_title="Irrigation Prediction System",
    page_icon="💧",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .recommendation-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def try_load_model():
    """Try multiple methods to load the model"""
    model_path = 'best_irrigation_model.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in the current directory.")
        return None
    
    st.info(f"Found model file: {model_path} (Size: {os.path.getsize(model_path)} bytes)")
    
    # Method 1: Try joblib first (more reliable for scikit-learn models)
    try:
        st.write("🔄 Attempting to load with joblib...")
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully with joblib!")
        return model
    except Exception as e:
        st.warning(f"Joblib loading failed: {str(e)}")
    
    # Method 2: Try pickle with different protocols
    try:
        st.write("🔄 Attempting to load with pickle...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully with pickle!")
        return model
    except Exception as e:
        st.warning(f"Pickle loading failed: {str(e)}")
    
    # Method 3: Try specific encoding
    try:
        st.write("🔄 Attempting to load with latin1 encoding...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        st.success("✅ Model loaded successfully with latin1 encoding!")
        return model
    except Exception as e:
        st.warning(f"Latin1 encoding failed: {str(e)}")
    
    return None

def create_demo_model():
    """Create a simple demo model if the main model fails to load"""
    st.info("🔄 Creating a demo model for demonstration purposes...")
    
    # Create a simple pipeline that works with HistGradientBoostingRegressor
    numeric_features = ['Soil_Moisture', 'Temperature', 'Humidity']
    categorical_features = ['Soil_Type', 'Crop_Type']
    
    numeric_transformer = StandardScaler()
    
    # Set sparse_output=False to get dense arrays for HistGradientBoostingRegressor
    categorical_transformer = OneHotEncoder(
        categories=[
            ['Clay', 'Loam', 'Sandy', 'Silt'],
            ['Barley', 'Cotton', 'Groundnut', 'Maize', 'Millet', 'Pulses', 'Rice', 'Soybean', 'Sugarcane', 'Wheat']
        ],
        handle_unknown='ignore',
        sparse_output=False  # This is the key fix
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create a simple model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(
            max_iter=50,  # Reduced for faster training
            random_state=42,
            max_depth=3
        ))
    ])
    
    # Create some dummy training data to fit the model
    np.random.seed(42)
    n_samples = 500  # Reduced for faster training
    
    dummy_data = pd.DataFrame({
        'Soil_Moisture': np.random.uniform(0, 100, n_samples),
        'Temperature': np.random.uniform(10, 40, n_samples),
        'Humidity': np.random.uniform(20, 90, n_samples),
        'Soil_Type': np.random.choice(['Clay', 'Loam', 'Sandy', 'Silt'], n_samples),
        'Crop_Type': np.random.choice(['Maize', 'Wheat', 'Rice', 'Cotton', 'Soybean'], n_samples)
    })
    
    # Create dummy target based on sensible rules
    soil_type_weights = {'Clay': 0.8, 'Loam': 1.0, 'Sandy': 1.2, 'Silt': 0.9}
    crop_type_weights = {
        'Rice': 1.5, 'Sugarcane': 1.4, 'Cotton': 1.3, 'Maize': 1.2, 
        'Soybean': 1.1, 'Wheat': 1.0, 'Barley': 0.9, 'Groundnut': 0.8, 
        'Pulses': 0.7, 'Millet': 0.6
    }
    
    dummy_target = (
        (100 - dummy_data['Soil_Moisture']) * 0.5 +  # Dry soil needs more water
        (dummy_data['Temperature'] - 20) * 0.8 +      # Higher temp needs more water
        (100 - dummy_data['Humidity']) * 0.3 +        # Lower humidity needs more water
        dummy_data['Soil_Type'].map(soil_type_weights) * 10 +
        dummy_data['Crop_Type'].map(crop_type_weights) * 15 +
        np.random.normal(0, 5, n_samples)  # Reduced noise
    )
    
    # Ensure target is positive
    dummy_target = np.maximum(dummy_target, 10)
    
    model.fit(dummy_data, dummy_target)
    st.warning("⚠️ Using demo model - predictions are for demonstration only")
    return model

def create_simple_demo_model():
    """Create an even simpler demo model as fallback"""
    st.info("🔄 Creating simple demo model...")
    
    # Simple rule-based prediction without ML
    class SimpleIrrigationModel:
        def predict(self, X):
            predictions = []
            for _, row in X.iterrows():
                soil_moisture = row['Soil_Moisture']
                temperature = row['Temperature']
                humidity = row['Humidity']
                soil_type = row['Soil_Type']
                crop_type = row['Crop_Type']
                
                # Simple rule-based calculation
                base_score = (100 - soil_moisture) * 0.3
                temp_score = max(0, temperature - 20) * 0.5
                humidity_score = (100 - humidity) * 0.2
                
                # Soil type factors
                soil_factors = {'Clay': 0.8, 'Loam': 1.0, 'Sandy': 1.3, 'Silt': 0.9}
                soil_score = soil_factors.get(soil_type, 1.0) * 10
                
                # Crop type factors
                crop_factors = {
                    'Rice': 1.6, 'Sugarcane': 1.5, 'Cotton': 1.4, 'Maize': 1.3,
                    'Soybean': 1.2, 'Wheat': 1.1, 'Barley': 1.0, 'Groundnut': 0.9,
                    'Pulses': 0.8, 'Millet': 0.7
                }
                crop_score = crop_factors.get(crop_type, 1.0) * 15
                
                prediction = base_score + temp_score + humidity_score + soil_score + crop_score
                predictions.append(prediction)
            
            return np.array(predictions)
    
    return SimpleIrrigationModel()

def main():
    st.markdown('<div class="main-header">🌱 Smart Irrigation Prediction System</div>', unsafe_allow_html=True)
    
    # Model loading section
    with st.expander("Model Status", expanded=True):
        model = try_load_model()
        
        if model is None:
            st.error("""
            **Could not load the original model file.**
            
            Using a demo model for demonstration purposes.
            """)
            try:
                model = create_demo_model()
            except Exception as e:
                st.warning(f"Demo model creation failed: {e}. Using simple rule-based model.")
                model = create_simple_demo_model()
        else:
            st.success("**Original model loaded successfully!**")
    
    # Input section
    st.markdown("### 📊 Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Environmental Conditions")
        
        soil_moisture = st.slider(
            "**Soil Moisture (%)**", 
            min_value=0.0, 
            max_value=100.0, 
            value=45.0,
            help="Current moisture level in the soil"
        )
        
        temperature = st.slider(
            "**Temperature (°C)**", 
            min_value=5.0, 
            max_value=45.0, 
            value=28.0,
            help="Current ambient temperature"
        )
        
        humidity = st.slider(
            "**Humidity (%)**", 
            min_value=10.0, 
            max_value=95.0, 
            value=65.0,
            help="Current relative humidity"
        )
    
    with col2:
        st.subheader("Crop & Soil Information")
        
        soil_type = st.selectbox(
            "**Soil Type**",
            ["Clay", "Loam", "Sandy", "Silt"],
            index=1,
            help="Type of soil in the field"
        )
        
        crop_type = st.selectbox(
            "**Crop Type**",
            ["Barley", "Cotton", "Groundnut", "Maize", "Millet", 
             "Pulses", "Rice", "Soybean", "Sugarcane", "Wheat"],
            index=3,
            help="Type of crop being cultivated"
        )
    
    # Prediction section
    st.markdown("---")
    
    if st.button("🚀 Predict Irrigation Need", type="primary", use_container_width=True):
        # Create input data
        input_data = pd.DataFrame({
            'Soil_Moisture': [soil_moisture],
            'Temperature': [temperature],
            'Humidity': [humidity],
            'Soil_Type': [soil_type],
            'Crop_Type': [crop_type]
        })
        
        try:
            # Make prediction
            with st.spinner("Analyzing conditions and predicting irrigation needs..."):
                prediction = model.predict(input_data)[0]
            
            # Display results
            st.markdown("### 📈 Prediction Results")
            
            # Main prediction card
            with st.container():
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="**Predicted Irrigation Need**", 
                        value=f"{prediction:.1f} units",
                        delta=None,
                        help="Higher values indicate greater irrigation requirement"
                    )
                
                with col2:
                    # Interpret the prediction
                    if prediction < 30:
                        status = "Low"
                        color = "🟢"
                        level = "success"
                    elif prediction < 60:
                        status = "Moderate"
                        color = "🟡"
                        level = "warning"
                    else:
                        status = "High"
                        color = "🔴"
                        level = "error"
                    
                    st.metric(label="**Irrigation Level**", value=f"{color} {status}")
                
                with col3:
                    # Water requirement indicator
                    water_req = min(100, (prediction / 80) * 100)
                    st.metric(
                        label="**Water Requirement**", 
                        value=f"{water_req:.1f}%",
                        delta=None
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                st.markdown("**🌊 Irrigation Advice:**")
                
                if prediction < 30:
                    st.write("• Minimal irrigation needed")
                    st.write("• Monitor soil moisture regularly")
                    st.write("• Consider natural rainfall sufficient")
                elif prediction < 60:
                    st.write("• Standard irrigation recommended")
                    st.write("• Water during cooler hours")
                    st.write("• Check soil before next irrigation")
                else:
                    st.write("• Significant irrigation required")
                    st.write("• Consider multiple watering sessions")
                    st.write("• Monitor for water stress signs")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with rec_col2:
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                st.markdown("**🌱 Crop-Specific Tips:**")
                
                crop_tips = {
                    "Rice": "Maintain consistent water level in fields",
                    "Wheat": "Moderate water during growth stages",
                    "Maize": "Regular irrigation during tasseling",
                    "Cotton": "Careful water management during boll formation",
                    "Sugarcane": "High water requirement throughout growth",
                    "Barley": "Moderate irrigation, drought tolerant",
                    "Pulses": "Low to moderate water requirements",
                    "Soybean": "Regular irrigation during pod formation",
                    "Groundnut": "Moderate water, avoid waterlogging",
                    "Millet": "Drought tolerant, minimal irrigation"
                }
                
                tip = crop_tips.get(crop_type, "Regular monitoring recommended")
                st.write(f"• {tip}")
                st.write("• Adjust based on growth stage")
                st.write("• Consider local weather forecasts")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Water conservation section
            st.markdown("### 💧 Water Conservation Tips")
            
            tips_col1, tips_col2, tips_col3 = st.columns(3)
            
            with tips_col1:
                st.write("**Efficient Irrigation**")
                st.write("• Use drip irrigation systems")
                st.write("• Water during early morning")
                st.write("• Avoid watering during wind")
            
            with tips_col2:
                st.write("**Soil Management**")
                st.write("• Use organic mulch")
                st.write("• Improve soil organic matter")
                st.write("• Practice conservation tillage")
            
            with tips_col3:
                st.write("**Monitoring**")
                st.write("• Use soil moisture sensors")
                st.write("• Monitor weather forecasts")
                st.write("• Regular field inspections")
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Try using different input values or check the model configuration.")

    # Sidebar information
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        This system predicts irrigation needs using machine learning.
        
        **Input Parameters:**
        - Soil Moisture (%)
        - Temperature (°C)
        - Humidity (%)
        - Soil Type
        - Crop Type
        
        **Output:**
        - Irrigation requirement in relative units
        - Conservation recommendations
        - Crop-specific advice
        """)
        
        st.markdown("## 📊 Model Info")
        if model:
            st.write(f"**Model Type**: {type(model).__name__}")
        
        st.markdown("## 💡 Tips")
        st.write("• Adjust inputs based on real-time field conditions")
        st.write("• Consider local weather forecasts")
        st.write("• Regular monitoring improves accuracy")
        st.write("• Combine with soil sensor data for best results")

if __name__ == "__main__":
    main()