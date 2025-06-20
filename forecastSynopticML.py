# Importação das dependencias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import joblib
import json
import warnings
import os
warnings.filterwarnings('ignore')

class SynopticMLForecast:
    def __init__(self):
        """
        Sistema de previsão meteorológica usando Machine Learning
        """
        self.models = {}
        self.scalers = {}
        self.features_names = []
        self.teleconnnection_indices = {}
    
    def generate_synthetic_weather_data(self, n_samples=5000, n_locations=20):
        """
        Gera dados sintéticos meteorológicos
        """
        # Gera dados sintéticos meteorológicos
        np.random.seed(42)

        # Coordenadas das estações (Brasil)
        lats = np.random.uniform(-33, 5, n_locations)
        lons = np.random.uniform(-73, -34, n_locations)

        data = []

        for i in range(n_samples):
            # Data base
            base_date = datetime(2020, 1, 1) + timedelta(days=i % 1460) # 4 anos

            # Variaveis temporais
            day_of_year = base_date.timetuple().tm_yday
            month = base_date.month
            season = (month % 12 + 3) // 3 # 1=Winter, 2=Spring, 3=Summer, 4=Fall

            # Teleconexões simukadas
            """
            teleconnections = {
                "ENSO": np.sin(2 * np.pi * day_of_year / 365) + 0.5 * np.random.normal(),
                "NAO": np.cos(2 * np.pi * day_of_year / 365) + 0.3 * np.random.normal()
            }"""
            enso_index = np.sin(2 * np.pi * day_of_year / 365) + 0.5 * np.random.normal()
            nao_index = np.cos(2 * np.pi * day_of_year / 365) + 0.3 * np.random.normal()

            for loc_idx in range(n_locations):
                lat, lon = lats[loc_idx], lons[loc_idx]

                # Temperatura base com sazonalidade e localização
                temp_base = 25 - 0.6 * lat + 8 * np.sin(2 * np.pi * day_of_year / 365)
                temp_enso_effect = 2 * enso_index if lat < -10 else 1 * enso_index
                temperature = temp_base + temp_enso_effect + np.random.normal(0, 2)

                # Pressão atmosferica
                pressure_base = 1013 + 10 * np.sin(np.radians(lat))
                pressure_seasonal = 5 * np.cos(2 * np.pi * day_of_year / 365)
                pressure = pressure_base + pressure_seasonal + np.random.normal(0, 3)

                # Umidade relativa
                humidity_base = 70 + 10 * np.sin(np.radians(lat+30))
                humidity_seasonal = 15 * np.sin(2 * np.pi * (day_of_year + 90) / 365)
                humidity = humidity_base + humidity_seasonal + np.random.normal(0, 5)
                humidity = np.clip(humidity, 30, 95)

                # Precipitação (correlacioada com umidade e ENSO)
                precip_prob = (humidity - 30) / 65 * 0.3 + 0.1 * enso_index
                precipitation = np.random.exponential(5) if np.random.random() < precip_prob else 0

                # Vento
                wind_speed = 3 + 2 * np.abs(nao_index) + np.random.exponential(2)
                wind_direction = np.random.uniform(0, 360)

                # Variaveis derivadas

                dewpoint = temperature - ((100 - humidity) / 5)
                heat_index = self.calculate_heat_index(temperature, humidity)

                # Padrões sinóticos simulados
                gradient_pressure = np.random.normal(0, 2) # Gradiente de pressão
                vorticity = np.random.normal(0, 1) # Vorticidade

                data.append({
                    'date': base_date,
                    'location_id': loc_idx,
                    'lat': lat,
                    'lon': lon,
                    'day_of_year': day_of_year,
                    'month': month,
                    'season': season,
                    'temperature': temperature,
                    'pressure': pressure,
                    'humidity': humidity,
                    'precipitation': precipitation,
                    'wind_speed': wind_speed,
                    'wind_direction': wind_direction,
                    'dewpoint': dewpoint,
                    'heat_index': heat_index,
                    'gradient_pressure': gradient_pressure,
                    'vorticity': vorticity,
                    # targets (proximos 3 dias)
                    'temp_1d': temperature + np.random.normal(0, 1),
                    'temp_3d': temperature + np.random.normal(0, 2),
                    'pressure_1d': pressure + np.random.normal(0, 2),
                    'pressure_3d': pressure + np.random.normal(0, 3),
                    'precipitation_1d': precipitation + np.random.normal(0, 2),
                    'precipitation_3d': precipitation + np.random.normal(0, 3)
                })

        return pd.DataFrame(data)
    
    def calculate_heat_index(self, temp, humidity):
        """
        Calcula índice de calor
        """
        if temp < 27:
            return temp
        
        hi = (0.5 * (temp + 61 + ((temp - 68) * 1.2) + (humidity * 0.094)))

        if hi > 79:
            hi = (-42.379 + 2.04901523 * temp + 10.14333127 * humidity - 
                0.22475541 * temp * humidity - 6.83783e-3 * temp**2 - 
                5.481717e-2 * humidity**2 + 1.22874e-3 * temp**2 * humidity + 
                8.5282e-4 * temp * humidity**2 - 1.99e-6 * temp**2 * humidity**2)

        return hi
    
    def create_features(self, df):
        """
        Cria features para modelos de ML
        """
        df_features = df.copy()

        # Features temporais
        df_features['sin_day'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
        df_features['cos_day'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)

        # Features de localização
        df_features['lat_sin'] = np.sin(np.radians(df_features['lat']))
        df_features['lon_sin'] = np.sin(np.radians(df_features['lon']))

        # Features de interação
        df_features['temp_humidity'] = df_features['temperature'] * df_features['humidity']
        df_features['pressure_gradient_abs'] = np.abs(df_features['gradient_pressure'])

        # Log features

        return df_features
    
    def prepare_lstm_data(self, df, sequence_lenght=7, target_col='temp_1d'):
        """
        Prepara dados para modelo LSTM
        """
        sequences = []
        targets = []

        features_cols = ['temperature', 'pressure', 'humidity', 'precipitation', 'wind_speed', 'enso_index', 'nao_index', 'sin_day', 'cos_day']

        for location in df['location_id'].unique():
            location_data = df[df['location_id'] == location].sort_values('date')

            for i in range(len(location_data) - sequence_lenght):
                seq = location_data[feature_cols].iloc[i:i+sequence_length].values
                target = location_data[target_col].iloc[i+sequence_length]

                sequences.append(seq)
                targets.append(target)

        return np.array(sequences), np.array(targets)
        


                
