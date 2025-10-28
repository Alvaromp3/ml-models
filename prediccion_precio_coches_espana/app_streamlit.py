import streamlit as st
import pandas as pd
from modelo import entrenar_modelo, predecir
import os

st.set_page_config(
    page_title="Used Car Price Predictor | Alvaro Martin-Pena",
    page_icon="🚗",
    layout="wide"
)

CSS = """
<style>
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #2563eb;
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
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
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
        border-left: 4px solid #2563eb;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

DATASET_PATH = os.getenv('CARS_DATASET_PATH', 'coches_espana_usados_1500_edad.csv')

def main():
    st.markdown('<h1 class="main-header">🚗 Used Car Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="author-info">Developed by Alvaro Martin-Pena | Machine Learning Engineer</p>', unsafe_allow_html=True)
    
    st.header("Car Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Technical Specifications")
        edad = st.number_input("Age (years)", min_value=0, max_value=50, value=5)
        kilometraje_km = st.number_input("Mileage (km)", min_value=0, value=50000)
        potencia_cv = st.number_input("Horsepower (CV)", min_value=0, value=100)
        consumo_l_100km = st.number_input("Consumption (L/100km)", min_value=0.0, value=6.5, format="%.1f")
        
        st.subheader("Vehicle Details")
        marca = st.text_input("Brand", value="Renault")
        modelo = st.text_input("Model", value="Clio")
    
    with col2:
        st.subheader("Ownership & Physical")
        num_duenos = st.number_input("Number of Owners", min_value=0, max_value=10, value=1)
        puertas = st.number_input("Number of Doors", min_value=2, max_value=5, value=5)
        
        st.subheader("Categorical Features")
        combustible = st.selectbox("Fuel Type", ["Gasolina", "Diesel", "Híbrido", "Eléctrico"])
        transmision = st.selectbox("Transmission", ["Manual", "Automática"])
        estado = st.selectbox("Condition", ["Excelente", "Bueno", "Regular", "Necesita Reparación"])
        region = st.selectbox("Region", ["Madrid", "Barcelona", "Valencia", "Sevilla", "Bilbao", "Otro"])
        color = st.selectbox("Color", ["Blanco", "Negro", "Gris", "Azul", "Rojo", "Otro"])
        carroceria = st.selectbox("Body Type", ["Sedán", "SUV", "Hatchback", "Berlina", "Otro"])
    
    if st.button("Train Model", type="primary", use_container_width=True):
        try:
            if not os.path.exists(DATASET_PATH):
                st.error(f"Dataset not found: {DATASET_PATH}")
                st.info("Set CARS_DATASET_PATH environment variable or place the CSV in the project directory")
                return
            
            with st.spinner("Training model..."):
                metrics_dict = entrenar_modelo(pd.read_csv(DATASET_PATH))
            
            st.success("Model trained successfully!")
            
            st.subheader("Test Set Metrics (Generalization)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score (Test)", f"{metrics_dict['test_r2']:.4f}")
            with col2:
                st.metric("MAE (Test)", f"€{metrics_dict['test_mae']:.2f}")
            with col3:
                st.metric("RMSE (Test)", f"€{metrics_dict['test_rmse']:.2f}")
            
            st.subheader("Train Set Metrics")
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("R² Score (Train)", f"{metrics_dict['train_r2']:.4f}")
            with col5:
                st.metric("MAE (Train)", f"€{metrics_dict['train_mae']:.2f}")
            with col6:
                st.metric("RMSE (Train)", f"€{metrics_dict['train_rmse']:.2f}")
            
            overfitting_diff = abs(metrics_dict['train_r2'] - metrics_dict['test_r2'])
            if overfitting_diff > 0.15:
                st.warning(f"⚠️ Possible overfitting: R² difference = {overfitting_diff:.3f}")
            elif overfitting_diff > 0.05:
                st.info(f"✓ Low overfitting: R² difference = {overfitting_diff:.3f}")
            else:
                st.success(f"✓ No overfitting: R² difference = {overfitting_diff:.3f}")
            
            st.session_state['model_trained'] = True
            st.session_state['metrics'] = metrics_dict
        
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
    
    st.markdown("---")
    
    if st.button("Predict Price", type="secondary", use_container_width=True):
        try:
            resultado = predecir(
                edad, kilometraje_km, potencia_cv, consumo_l_100km, num_duenos, puertas,
                marca, modelo, combustible, transmision, estado, region, color, carroceria
            )
            
            if isinstance(resultado, str):
                st.error(resultado)
            else:
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>Predicted Price</h2>
                    <h1 style="font-size: 4rem; margin: 1rem 0;">€{resultado:,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem; background: rgba(37, 99, 235, 0.1); border-radius: 15px; margin-top: 3rem;">
        <p style="font-size: 1.5rem; font-weight: 600; color: #2563eb; margin-bottom: 0.5rem;">🚗 Used Car Price Predictor</p>
        <p><strong>Developed by Alvaro Martin-Pena</strong></p>
        <p>Machine Learning Engineer | Data Scientist</p>
        <p style="font-size: 0.9rem; margin-top: 1rem; opacity: 0.8;">© 2024 Alvaro Martin-Pena - All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

