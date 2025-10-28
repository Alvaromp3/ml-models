import os
import streamlit as st
from modelo import entrenar_modelo, predecir

def get_explanation(data, quality):
    if "Alta" in quality or "High" in quality:
        return "✅ High quality wines (≥6) typically have balanced acidity, moderate alcohol content, and good structure."
    else:
        return "⚠️ Low quality wines (<6) often have imbalanced characteristics, such as high volatile acidity or low alcohol content."

st.set_page_config(
    page_title="Wine Quality Predictor | Developed by Alvaro Martin-Pena",
    page_icon="🍷",
    layout="centered"
)

st.markdown("""
<div style="text-align:center">
  <h1 style="margin-bottom:4px">🍷 Wine Quality Predictor</h1>
  <p style="color:#6c757d;margin-top:0">Developed by <strong>Alvaro Martin-Pena</strong> · Machine Learning Engineer</p>
</div>
""", unsafe_allow_html=True)

with st.expander("Dataset configuration", expanded=False):
    st.write("Use env var `WINE_DATASET_PATH` to set a custom dataset path.")
    st.code(os.getenv('WINE_DATASET_PATH', 'winequalityN.csv'))

st.subheader("Model training and metrics")
left, right = st.columns([1,2])
with left:
    if st.button("Train model", use_container_width=True):
        try:
            metrics = entrenar_modelo()
            st.success("Model trained successfully!")
            with right:
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Train Acc", f"{metrics['train_accuracy']:.3f}")
                t2.metric("Train Prec", f"{metrics['train_precision']:.3f}")
                t3.metric("Train Rec", f"{metrics['train_recall']:.3f}")
                t4.metric("Train F1", f"{metrics['train_f1']:.3f}")

                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Test Acc", f"{metrics['test_accuracy']:.3f}")
                b2.metric("Test Prec", f"{metrics['test_precision']:.3f}")
                b3.metric("Test Rec", f"{metrics['test_recall']:.3f}")
                b4.metric("Test F1", f"{metrics['test_f1']:.3f}")

                gap = metrics['train_accuracy'] - metrics['test_accuracy']
                if gap > 0.15:
                    st.warning(f"Possible overfitting detected (Accuracy gap = {gap:.3f}).")
                elif gap > 0.05:
                    st.info(f"Small train/test gap (Accuracy gap = {gap:.3f}).")
                else:
                    st.success(f"Good generalization (Accuracy gap = {gap:.3f}).")
        except Exception as e:
            st.error(f"Error training model: {e}")

st.subheader("Predict quality")
with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    with c1:
        fixed_acidity = st.number_input("Fixed Acidity", value=7.4, step=0.1)
        volatile_acidity = st.number_input("Volatile Acidity", value=0.7, step=0.01)
        citric_acid = st.number_input("Citric Acid", value=0.0, step=0.01)
        residual_sugar = st.number_input("Residual Sugar", value=1.9, step=0.1)
        chlorides = st.number_input("Chlorides", value=0.076, step=0.001)
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0, step=1.0)
    with c2:
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0, step=1.0)
        density = st.number_input("Density", value=0.9978, step=0.0001, format="%.4f")
        ph = st.number_input("pH", value=3.51, step=0.01)
        sulphates = st.number_input("Sulphates", value=0.56, step=0.01)
        alcohol = st.number_input("Alcohol", value=9.4, step=0.1)
        wine_type = st.selectbox("Type", options=["red", "white"], index=0)

    submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        data = {
            'fixed acidity': float(fixed_acidity),
            'volatile acidity': float(volatile_acidity),
            'citric acid': float(citric_acid),
            'residual sugar': float(residual_sugar),
            'chlorides': float(chlorides),
            'free sulfur dioxide': float(free_sulfur_dioxide),
            'total sulfur dioxide': float(total_sulfur_dioxide),
            'density': float(density),
            'pH': float(ph),
            'sulphates': float(sulphates),
            'alcohol': float(alcohol),
            'type': wine_type
        }
        quality, confidence = predecir(data)
        if quality is None:
            st.error(confidence)
        else:
            if "Alta" in quality:
                st.success("High quality (≥ 6)")
            else:
                st.error("Low quality (< 6)")
            st.caption(confidence)
            st.info(get_explanation(data, quality))

st.markdown("---")
st.caption("Developed by Alvaro Martin-Pena")


