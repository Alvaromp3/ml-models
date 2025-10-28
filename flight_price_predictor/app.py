import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="✈️ Iberia Flight Price Predictor",
    page_icon="✈️",
    layout="wide"
)

# CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    }
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        color: white !important;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.4);
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        color: rgba(255,255,255,0.95);
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin: 1.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .price-display {
        font-size: 4rem;
        font-weight: 800;
        color: white;
        text-align: center;
        margin: 2rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1.2rem 2.5rem;
        font-size: 1.2rem;
        font-weight: 700;
        width: 100%;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(59, 130, 246, 0.6);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        color: white;
    }
    
    .flight-info {
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.1) 100%);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 1rem;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .input-section {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Airport coordinates
AIRPORTS = {
    'Madrid': {'lat': 40.4719, 'lon': -3.5626, 'code': 'MAD'},
    'Barcelona': {'lat': 41.2974, 'lon': 2.0833, 'code': 'BCN'},
    'Valencia': {'lat': 39.4893, 'lon': -0.4816, 'code': 'VLC'},
    'Sevilla': {'lat': 37.4180, 'lon': -5.8931, 'code': 'SVQ'},
    'Bilbao': {'lat': 43.3011, 'lon': -2.9106, 'code': 'BIO'},
    'Palma': {'lat': 39.5517, 'lon': 2.7388, 'code': 'PMI'},
    'Las Palmas': {'lat': 27.9319, 'lon': -15.3866, 'code': 'LPA'},
    'Malaga': {'lat': 36.6749, 'lon': -4.4991, 'code': 'AGP'},
    'Alicante': {'lat': 38.2822, 'lon': -0.5581, 'code': 'ALC'},
    'Santiago': {'lat': 42.8963, 'lon': -8.4151, 'code': 'SCQ'}
}

def limpiar_datos(df):
    df_clean = df.copy()
    
    print(f'El shape del df sin limpiar es {df.shape}')
    print('Estamos limpiado el df')
    
    # eliminamos duplicados 
    df_clean = df_clean.drop_duplicates()
    
    # no tenemos filas con NaN 
    numeric_cols = [col for col in df_clean.select_dtypes(include=['float64','int64']).columns]
    cat_cols = [col for col in df_clean.select_dtypes(include=['object']).columns]
    
    # eliminamos outliers con IQR 
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1 
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= limite_inferior) & (df_clean[col] <= limite_superior)]
    
    print(f'Hemos acabado de limpiar el df')
    print(f'El df limpio tiene {df_clean.shape[0]} filas y {df_clean.shape[1]} columnas')
    
    return df_clean

def entrenar_mejor_modelo(df, target_col):
    df_clean = limpiar_datos(df)
    
    # separamos en x , y
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    numeric_col = [col for col in df_clean.select_dtypes(include=['float64', 'int64']).columns if col != target_col]
    cat_col = [col for col in df_clean.select_dtypes(include=['object']).columns if col != target_col]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_col),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_col)
        ]
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # definir mejor modelo 
    model = CatBoostRegressor(verbose=0)
    
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # definir param_grid 
    param_grid = {
        'model__iterations': [300, 500],
        'model__depth': [6, 8],
        'model__learning_rate': [0.03, 0.05],
        'model__l2_leaf_reg': [3, 5]
    }
    
    # crear GridSearch para encontrar los mejores parametros
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=kf,
        scoring='r2',
        n_jobs=-1,  # para usar todos los nucleos de nuestra CPU
    )

    # entrenar grid
    grid.fit(X, y)
    
    # imprimir best_parametrers 
    print(f"\nBest Parameters:\n{grid.best_params_}")
    print(f"\nBest R² Score: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_

@st.cache_resource
def train_model():
    df = pd.read_csv('/Users/alvaromartin-pena/anaconda_projects/4e1a8735-845d-491d-84cd-848cf9c616ea/iberia_flight_prices_2025.csv')
    
    best_model, best_params = entrenar_mejor_modelo(df, 'price_eur')
    
    # Calculate metrics on test set
    df_clean = limpiar_datos(df)
    X = df_clean.drop(columns=['price_eur'])
    y = df_clean['price_eur']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return best_model, r2, mae, df_clean.columns.tolist()

def create_flight_path_animation(origin, destination):
    # Map real airport codes to coordinates
    airport_coords = {
        'MAD': {'lat': 40.4719, 'lon': -3.5626, 'name': 'Madrid'},
        'BCN': {'lat': 41.2974, 'lon': 2.0833, 'name': 'Barcelona'},
        'LIS': {'lat': 38.7813, 'lon': -9.1363, 'name': 'Lisboa'},
        'LHR': {'lat': 51.4700, 'lon': -0.4543, 'name': 'Londres'},
        'JFK': {'lat': 40.6413, 'lon': -73.7781, 'name': 'Nueva York'},
        'SFO': {'lat': 37.6213, 'lon': -122.3790, 'name': 'San Francisco'},
        'BOG': {'lat': 4.7016, 'lon': -74.1469, 'name': 'Bogotá'},
        'MIA': {'lat': 25.7959, 'lon': -80.2870, 'name': 'Miami'},
        'GRU': {'lat': -23.4356, 'lon': -46.4731, 'name': 'São Paulo'},
        'OPO': {'lat': 41.2482, 'lon': -8.6814, 'name': 'Oporto'},
        'BOS': {'lat': 42.3656, 'lon': -71.0096, 'name': 'Boston'},
        'FRA': {'lat': 50.0379, 'lon': 8.5622, 'name': 'Frankfurt'},
        'FCO': {'lat': 41.8003, 'lon': 12.2389, 'name': 'Roma'},
        'SCL': {'lat': -33.3927, 'lon': -70.7854, 'name': 'Santiago'},
        'MEX': {'lat': 19.4363, 'lon': -99.0721, 'name': 'Ciudad de México'},
        'LIM': {'lat': -12.0219, 'lon': -77.1143, 'name': 'Lima'},
        'EZE': {'lat': -34.8222, 'lon': -58.5358, 'name': 'Buenos Aires'},
        'LAX': {'lat': 33.9425, 'lon': -118.4081, 'name': 'Los Ángeles'},
        'AMS': {'lat': 52.3105, 'lon': 4.7683, 'name': 'Ámsterdam'},
        'CDG': {'lat': 49.0097, 'lon': 2.5479, 'name': 'París'},
        'ZRH': {'lat': 47.4647, 'lon': 8.5492, 'name': 'Zúrich'},
        'ATH': {'lat': 37.9364, 'lon': 23.9445, 'name': 'Atenas'}
    }
    
    origin_coords = airport_coords.get(origin, {'lat': 40.4719, 'lon': -3.5626, 'name': origin})
    dest_coords = airport_coords.get(destination, {'lat': 41.2974, 'lon': 2.0833, 'name': destination})
    
    # Create intermediate points for smooth animation
    n_points = 50
    lats = np.linspace(origin_coords['lat'], dest_coords['lat'], n_points)
    lons = np.linspace(origin_coords['lon'], dest_coords['lon'], n_points)
    
    # Create the map
    fig = go.Figure()
    
    # Add the flight path
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='lines',
        line=dict(width=3, color='red'),
        name='Ruta de Vuelo',
        hoverinfo='skip'
    ))
    
    # Add origin airport
    fig.add_trace(go.Scattermapbox(
        lat=[origin_coords['lat']],
        lon=[origin_coords['lon']],
        mode='markers+text',
        marker=dict(size=15, color='green'),
        text=[f"{origin_coords.get('name', origin)} ({origin})"],
        textposition="top center",
        name='Origen',
        hovertemplate=f"<b>{origin_coords.get('name', origin)}</b><br>Código: {origin}<extra></extra>"
    ))
    
    # Add destination airport
    fig.add_trace(go.Scattermapbox(
        lat=[dest_coords['lat']],
        lon=[dest_coords['lon']],
        mode='markers+text',
        marker=dict(size=15, color='blue'),
        text=[f"{dest_coords.get('name', destination)} ({destination})"],
        textposition="top center",
        name='Destino',
        hovertemplate=f"<b>{dest_coords.get('name', destination)}</b><br>Código: {destination}<extra></extra>"
    ))
    
    # Add animated airplane
    fig.add_trace(go.Scattermapbox(
        lat=[origin_coords['lat']],
        lon=[origin_coords['lon']],
        mode='markers',
        marker=dict(size=20, color='orange', symbol='airport'),
        name='Vuelo',
        hovertemplate="Vuelo en progreso<extra></extra>"
    ))
    
    # Calculate dynamic zoom based on distance
    lat_diff = abs(origin_coords['lat'] - dest_coords['lat'])
    lon_diff = abs(origin_coords['lon'] - dest_coords['lon'])
    max_diff = max(lat_diff, lon_diff)
    
    if max_diff < 5:
        zoom_level = 6
    elif max_diff < 20:
        zoom_level = 4
    elif max_diff < 50:
        zoom_level = 2
    else:
        zoom_level = 1
    
    # Configure map layout
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(
                lat=(origin_coords['lat'] + dest_coords['lat']) / 2,
                lon=(origin_coords['lon'] + dest_coords['lon']) / 2
            ),
            zoom=zoom_level
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
    )
    
    return fig

def main():
    st.markdown('<h1 class="main-header">✈️ Predictor de Precios Iberia</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predice precios de vuelos y visualiza tu viaje</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner('Cargando modelo de predicción...'):
        model, r2_score_val, mae_val, feature_cols = train_model()
    
    # Get available origins and destinations from the data
    df_sample = pd.read_csv('/Users/alvaromartin-pena/anaconda_projects/4e1a8735-845d-491d-84cd-848cf9c616ea/iberia_flight_prices_2025.csv')
    available_origins = sorted(df_sample['origin'].unique())
    available_destinations = sorted(df_sample['destination'].unique())
    
    # Input form in main area
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Detalles del Vuelo</h3>', unsafe_allow_html=True)
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    # Define calculate_distance function outside columns
    def calculate_distance(origin_code, dest_code):
        if origin_code == dest_code:
            return 0
        airport_coords = {
            'MAD': {'lat': 40.4719, 'lon': -3.5626},
            'BCN': {'lat': 41.2974, 'lon': 2.0833},
            'LIS': {'lat': 38.7813, 'lon': -9.1363},
            'LHR': {'lat': 51.4700, 'lon': -0.4543},
            'JFK': {'lat': 40.6413, 'lon': -73.7781},
            'SFO': {'lat': 37.6213, 'lon': -122.3790},
            'BOG': {'lat': 4.7016, 'lon': -74.1469},
            'MIA': {'lat': 25.7959, 'lon': -80.2870},
            'GRU': {'lat': -23.4356, 'lon': -46.4731},
            'OPO': {'lat': 41.2482, 'lon': -8.6814},
            'BOS': {'lat': 42.3656, 'lon': -71.0096},
            'FRA': {'lat': 50.0379, 'lon': 8.5622},
            'FCO': {'lat': 41.8003, 'lon': 12.2389},
            'SCL': {'lat': -33.3927, 'lon': -70.7854},
            'MEX': {'lat': 19.4363, 'lon': -99.0721},
            'LIM': {'lat': -12.0219, 'lon': -77.1143},
            'EZE': {'lat': -34.8222, 'lon': -58.5358},
            'LAX': {'lat': 33.9425, 'lon': -118.4081},
            'AMS': {'lat': 52.3105, 'lon': 4.7683},
            'CDG': {'lat': 49.0097, 'lon': 2.5479},
            'ZRH': {'lat': 47.4647, 'lon': 8.5492},
            'ATH': {'lat': 37.9364, 'lon': 23.9445}
        }
        
        origin_coord = airport_coords.get(origin_code, {'lat': 40.4719, 'lon': -3.5626})
        dest_coord = airport_coords.get(dest_code, {'lat': 41.2974, 'lon': 2.0833})
        
        from math import radians, cos, sin, asin, sqrt
        
        lat1, lon1 = radians(origin_coord['lat']), radians(origin_coord['lon'])
        lat2, lon2 = radians(dest_coord['lat']), radians(dest_coord['lon'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371
        
        return round(c * r, 1)

    with col_input1:
        origin = st.selectbox("Origen", available_origins)
        cabin_class = st.selectbox("Clase de Cabina", ['Economy', 'Business'])
    
    with col_input2:
        destination = st.selectbox("Destino", available_destinations)
        stops = st.selectbox("Escalas", ['Direct', '1 Stop'])
        
        # Calculate and display distance dynamically
        if origin != destination:
            distance_km = calculate_distance(origin, destination)
            st.markdown(f'<p style="color: white; font-size: 1.1rem; margin: 0.5rem 0;">Distancia: {distance_km:,} km</p>', unsafe_allow_html=True)
        else:
            distance_km = 0
            st.markdown('<p style="color: white; font-size: 1.1rem; margin: 0.5rem 0;">Distancia: - km</p>', unsafe_allow_html=True)
    
    with col_input3:
        day_of_week_names = {
            'Lunes': 1, 'Martes': 2, 'Miércoles': 3, 'Jueves': 4, 
            'Viernes': 5, 'Sábado': 6, 'Domingo': 7
        }
        departure_date = st.date_input("Fecha de Salida", value=pd.to_datetime('2025-01-01'))
        day_of_week = departure_date.weekday() + 1
        month = departure_date.month
        
        # Calculate demand based on multiple factors
        def calculate_demand(date, origin_code, dest_code, cabin_class, distance):
            import random
            random.seed(int(date.strftime('%Y%m%d')) + hash(origin_code + dest_code))
            
            month = date.month
            day_of_week = date.weekday()
            day = date.day
            
            # Base demand varies by route popularity
            popular_routes = ['MAD-BCN', 'BCN-MAD', 'MAD-LHR', 'LHR-MAD']
            route = f"{origin_code}-{dest_code}"
            
            if route in popular_routes:
                base_demand = 1.0 + random.uniform(0, 0.2)
            else:
                base_demand = 0.7 + random.uniform(0, 0.3)
            
            # Seasonal adjustments
            if month in [6, 7, 8]:  # Summer peak
                base_demand *= 1.3 + random.uniform(0, 0.2)
            elif month == 12:  # Christmas
                base_demand *= 1.4 + random.uniform(0, 0.15)
            elif month in [3, 4, 5]:  # Spring
                base_demand *= 1.1 + random.uniform(0, 0.1)
            elif month in [1, 2]:  # Low season
                base_demand *= 0.8 + random.uniform(0, 0.1)
            
            # Day of week impact
            if day_of_week in [4, 5]:  # Friday, Saturday
                base_demand *= 1.2
            elif day_of_week == 6:  # Sunday
                base_demand *= 1.15
            elif day_of_week in [0, 1]:  # Monday, Tuesday
                base_demand *= 0.85
            
            # Distance factor (longer flights = higher demand)
            if distance > 5000:
                base_demand *= 1.1
            elif distance < 1000:
                base_demand *= 0.95
            
            # Cabin class premium
            if cabin_class == 'Business':
                base_demand *= 1.25
            
            # Holiday periods
            if (month == 12 and day >= 20) or (month == 1 and day <= 7):
                base_demand *= 1.2
            elif month == 8 and day >= 15:
                base_demand *= 1.15
            
            return max(0.6, min(1.8, round(base_demand, 2)))
        
        # Calculate demand with all factors
        if origin != destination:
            demand = calculate_demand(departure_date, origin, destination, cabin_class, distance_km)
        else:
            demand = 1.0
        st.markdown(f'<p style="color: white; font-size: 1.1rem; margin: 0.5rem 0;">Factor de demanda: {demand:.2f}</p>', unsafe_allow_html=True)
        
        time_band_names = {
            'Mañana': 'Morning', 'Tarde': 'Afternoon', 'Noche': 'Evening', 'Madrugada': 'Night'
        }
        time_band_name = st.selectbox("Franja Horaria", list(time_band_names.keys()))
        time_band = time_band_names[time_band_name]
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if origin == destination:
        st.error("El origen y destino no pueden ser iguales!")
        return
    
    # Prediction button
    if st.button("Predecir Precio del Vuelo", type="primary"):
        # Calculate final values for prediction
        final_distance = calculate_distance(origin, destination) if origin != destination else 0
        final_demand = calculate_demand(departure_date, origin, destination, cabin_class, final_distance) if origin != destination else 1.0
        
        # Prepare input data
        input_data = pd.DataFrame({
            'origin': [origin],
            'destination': [destination],
            'distance_km': [final_distance],
            'day_of_week': [day_of_week],
            'month': [month],
            'demand': [final_demand],
            'cabin_class': [cabin_class],
            'stops': [stops],
            'time_band': [time_band]
        })
        
        # Make prediction
        predicted_price = model.predict(input_data)[0]
        
        st.markdown(f'<div class="price-display">€{predicted_price:.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: white; text-align: center; font-size: 1.2rem; margin-top: 1rem;">Precisión del Modelo (R²): {r2_score_val:.3f} | Error Medio: €{mae_val:.2f}</p>', unsafe_allow_html=True)
    
    # Large map container
    st.markdown('<h3 class="section-title">Ruta del Vuelo</h3>', unsafe_allow_html=True)
    
    # Create and display the flight path in full width
    fig = create_flight_path_animation(origin, destination)
    st.plotly_chart(fig, use_container_width=True)
    
    # Flight details below map
    airport_coords = {
        'MAD': {'lat': 40.4719, 'lon': -3.5626, 'name': 'Madrid'},
        'BCN': {'lat': 41.2974, 'lon': 2.0833, 'name': 'Barcelona'},
        'LIS': {'lat': 38.7813, 'lon': -9.1363, 'name': 'Lisboa'},
        'LHR': {'lat': 51.4700, 'lon': -0.4543, 'name': 'Londres'},
        'JFK': {'lat': 40.6413, 'lon': -73.7781, 'name': 'Nueva York'},
        'SFO': {'lat': 37.6213, 'lon': -122.3790, 'name': 'San Francisco'},
        'BOG': {'lat': 4.7016, 'lon': -74.1469, 'name': 'Bogotá'},
        'MIA': {'lat': 25.7959, 'lon': -80.2870, 'name': 'Miami'},
        'GRU': {'lat': -23.4356, 'lon': -46.4731, 'name': 'São Paulo'},
        'OPO': {'lat': 41.2482, 'lon': -8.6814, 'name': 'Oporto'},
        'BOS': {'lat': 42.3656, 'lon': -71.0096, 'name': 'Boston'},
        'FRA': {'lat': 50.0379, 'lon': 8.5622, 'name': 'Frankfurt'},
        'FCO': {'lat': 41.8003, 'lon': 12.2389, 'name': 'Roma'},
        'SCL': {'lat': -33.3927, 'lon': -70.7854, 'name': 'Santiago'},
        'MEX': {'lat': 19.4363, 'lon': -99.0721, 'name': 'Ciudad de México'},
        'LIM': {'lat': -12.0219, 'lon': -77.1143, 'name': 'Lima'},
        'EZE': {'lat': -34.8222, 'lon': -58.5358, 'name': 'Buenos Aires'},
        'LAX': {'lat': 33.9425, 'lon': -118.4081, 'name': 'Los Ángeles'},
        'AMS': {'lat': 52.3105, 'lon': 4.7683, 'name': 'Ámsterdam'},
        'CDG': {'lat': 49.0097, 'lon': 2.5479, 'name': 'París'},
        'ZRH': {'lat': 47.4647, 'lon': 8.5492, 'name': 'Zúrich'},
        'ATH': {'lat': 37.9364, 'lon': 23.9445, 'name': 'Atenas'}
    }
    
    origin_info = airport_coords.get(origin, {'name': origin})
    dest_info = airport_coords.get(destination, {'name': destination})
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown(f'''
        <div class="flight-info">
        <h4 style="color: white; margin-bottom: 1rem;">Información del Vuelo</h4>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0;"><strong>Origen:</strong> {origin_info.get('name', origin)} ({origin})</p>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0;"><strong>Destino:</strong> {dest_info.get('name', destination)} ({destination})</p>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0;"><strong>Distancia:</strong> {calculate_distance(origin, destination):,} km</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col_info2:
        st.markdown(f'''
        <div class="flight-info">
        <h4 style="color: white; margin-bottom: 1rem;">Detalles del Servicio</h4>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0;"><strong>Clase:</strong> {cabin_class}</p>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0;"><strong>Escalas:</strong> {stops}</p>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0;"><strong>Horario:</strong> {time_band_name}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col_info3:
        st.markdown(f'''
        <div class="flight-info">
        <h4 style="color: white; margin-bottom: 1rem;">Fecha y Demanda</h4>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0;"><strong>Fecha:</strong> {departure_date.strftime('%d/%m/%Y')}</p>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0;"><strong>Demanda:</strong> {calculate_demand(departure_date, origin, destination, cabin_class, calculate_distance(origin, destination)) if origin != destination else 1.0:.2f}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Additional information
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Información del Modelo</h3>', unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown('''
        <div class="metric-card">
        <h4>Tipo de Modelo</h4>
        <p style="font-size: 1.2rem; font-weight: 600;">CatBoost Regressor</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
        <h4>Puntuación R²</h4>
        <p style="font-size: 1.5rem; font-weight: 700; color: #4ade80;">{r2_score_val:.3f}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col5:
        st.markdown(f'''
        <div class="metric-card">
        <h4>Error Medio</h4>
        <p style="font-size: 1.5rem; font-weight: 700; color: #f87171;">€{mae_val:.2f}</p>
        </div>
        ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()