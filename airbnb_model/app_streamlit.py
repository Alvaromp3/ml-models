import streamlit as st
import pandas as pd
from modelo_airbnb import ModeloAirbnb
import os

st.set_page_config(
    page_title="Airbnb Price Predictor | Alvaro Martin-Pena",
    page_icon="🏠",
    layout="wide"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #FF5A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .author-info {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #FF5A5F 0%, #FFB4B8 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #FF5A5F;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    modelo = ModeloAirbnb()
    if not modelo.cargar_modelo():
        with st.spinner('Training model...'):
            metrics = modelo.entrenar_modelo()
            modelo.guardar_modelo()
            return modelo, metrics
    return modelo, modelo.metrics

def load_data_choices():
    df = pd.read_csv('airbnb_synthetic.csv')
    property_types = sorted(df['property_type'].unique())
    room_types = sorted(df['room_type'].unique())
    return property_types, room_types

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Airbnb Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="author-info">Developed by Alvaro Martin-Pena | Machine Learning Engineer</p>', unsafe_allow_html=True)
    
    # Load model
    modelo, metrics = load_model()
    property_types, room_types = load_data_choices()
    
    # Sidebar with model metrics
    with st.sidebar:
        st.header("Model Performance Metrics")
        st.markdown(f"""
        <div class="metric-card">
            <h3>R² Score</h3>
            <h2>{metrics.get('r2', 0):.4f}</h2>
            <p>Coefficient of Determination</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>MAE</h3>
            <h2>${metrics.get('mae', 0):.2f}</h2>
            <p>Mean Absolute Error</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>RMSE</h3>
            <h2>${metrics.get('rmse', 0):.2f}</h2>
            <p>Root Mean Squared Error</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("**Model**: Random Forest Regressor (150 trees)")
        st.info("**Training Data**: Synthetic Airbnb dataset")
    
    # Main content
    st.header("Property Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Location")
        latitude = st.number_input("Latitude", value=40.7128, format="%.6f")
        longitude = st.number_input("Longitude", value=-74.0060, format="%.6f")
        
        st.subheader("Property Details")
        property_type = st.selectbox("Property Type", property_types)
        room_type = st.selectbox("Room Type", room_types)
    
    with col2:
        st.subheader("Features")
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        
        st.subheader("Reviews & Availability")
        number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=50)
        availability_365 = st.number_input("Days Available per Year", min_value=0, max_value=365, value=200)
    
    # Prediction button
    if st.button("Predict Price", type="primary", use_container_width=True):
        try:
            datos = {
                'latitude': latitude,
                'longitude': longitude,
                'property_type': property_type,
                'room_type': room_type,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'number_of_reviews': number_of_reviews,
                'availability_365': availability_365
            }
            
            prediccion = modelo.predecir(datos)
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-card">
                <h2>Predicted Price</h2>
                <h1 style="font-size: 4rem; margin: 1rem 0;">${prediccion:,.2f}</h1>
                <p style="font-size: 1.2rem;">per night</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature importance
            st.subheader("Feature Importance")
            if hasattr(modelo.pipeline.named_steps['model'], 'feature_importances_'):
                feature_names = ['latitude', 'longitude', 'bedrooms', 'bathrooms', 
                               'number_of_reviews', 'availability_365', 'property_type', 'room_type']
                importances = modelo.pipeline.named_steps['model'].feature_importances_
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                st.bar_chart(importance_df.set_index('Feature'))
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Airbnb Price Predictor</strong></p>
        <p>Developed by Alvaro Martin-Pena</p>
        <p>Machine Learning Engineer | Data Scientist</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">© 2024 Alvaro Martin-Pena - All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

