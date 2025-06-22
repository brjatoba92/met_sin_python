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

    
    def train_essemble_models(self, df, target_variable=None):
        """
        Treina ensemble de modelos
        """
        if target_variables is None:
            target_variables = ['temp_1d', 'temp_3d', 'pressure_1d', 'pressure_3d', 'precipitation_1d', 'precipitation_3d'] 
        
        # Preparar features
        df_features = self.create_features(df)

        # Selecionar features
        df_features = self.create_features(df)

        # Selecionar features para treinamento
        feature_cols = [col for col in df_features.columns 
                    if col not in target_variables + ['date', 'location_id']]
        
        self.features_names = feature_cols
        X = df_features[feature_cols]

        results = {}

        for target in target_variables:
            print(f"Treinando modelos para {target} ...")

            y = df_features[target]

            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Escalar dados
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            self.scalers[target] = scaler

            # Modelos
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),  
                'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            }

            target_results = {}

            for name, model in models.items():
                if name == 'MLP':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Metricas
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                target_results[name] = {
                    'model': model,
                    'mae': mae,
                    'mse': mse,
                    'r2': r2,
                    'predictions': y_pred,
                    'actual': y_test.values
                }

                print(f"  {name} - MAE: {mae:3f}, R2: {r2:3f}")

            # Ensemble (média ponderada baseada no R2)
            weights = {name: max(0, results['r2']) for name, results in target_results.items()}
            total_weight = sum(weights.values())

            if total_weight > 0:
                weights = {name: w/total_weight for name, w in weights.items()}

                ensemble_pred = sum(weights[name] * target_results[name]['predictions'] 
                                    for name in target_results.keys())

                ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                ensemble_r2 = r2_score(y_test, ensemble_pred)

                target_results['Ensemble'] = {
                    'model': None,
                    'mae': ensemble_mae,
                    'r2': ensemble_r2,
                    'predictions': ensemble_pred,
                    'actual': y_test.values
                }

                print(f"  Ensemble - MAE: {ensemble_mae:3f}, R2: {ensemble_r2:3f}")

            results[target] = target_results
            self.models[target] = target_results
        
        return results

    def train_lstm_model(self, df, target_col='temp_1d', sequence_lenght=7):
        """
        treina modelo LSTM para previsão de séries temporais
        """
        print(f"Treinando modelo LSTM para {target_col} ...")

        # Preparar dados
        df_features = self.create_features(df)
        X, y = self.prepare_lstm_data(df_features, sequence_lenght, target_col)

        # Dividir dados
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Escalar dados
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_train_scaled = scaler_X.reshape(X_train.shape)

        X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1]))
        X_test_scaled = scaler_X.reshape(X_test.shape)

        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        # Construir modelo LSTM
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=(sequence_lenght, X_train.shape[2])),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Treinar modelo
        history = model.fit(
            X_train_scaled, y_train_scaled,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            verbose=0
        )

        # Avaliar modelo
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f" LSTM - MAE: {mae:.3f}, R2: {r2:.3f}")

        # Salvar modelo e scalers
        lstm_results = {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'mae': mae,
            'r2': r2,
            'history': history.history,
            'predictions': y_pred,
            'actual': y_test
        }

        self.models[f'{target_col}_lstm'] = lstm_results

        return lstm_results