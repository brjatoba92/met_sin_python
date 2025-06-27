# Plataforma de Previsão Sinótica com Machine Learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
from flask import Flask, request, render_template_string, jsonify
warnings.filterwarnings('ignore')

class SynopticMLForecast:
    def __init__(self):
        """
        Sistema de previsão meteorológica usando Machine Learning
        """
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.teleconnection_indices = {}
        
    def generate_synthetic_weather_data(self, n_samples=5000, n_locations=20):
        """
        Gera dataset sintético de dados meteorológicos históricos
        """
        np.random.seed(42)
        
        # Coordenadas das estações (Brasil)
        lats = np.random.uniform(-33, 5, n_locations)
        lons = np.random.uniform(-73, -34, n_locations)
        
        data = []
        
        for i in range(n_samples):
            # Data base
            base_date = datetime(2020, 1, 1) + timedelta(days=i % 1460)  # 4 anos
            
            # Variáveis temporais
            day_of_year = base_date.timetuple().tm_yday
            month = base_date.month
            season = (month % 12 + 3) // 3  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
            
            # Teleconexões simuladas
            enso_index = np.sin(2 * np.pi * day_of_year / 365) + 0.5 * np.random.normal()
            nao_index = np.cos(2 * np.pi * day_of_year / 365) + 0.3 * np.random.normal()
            
            for loc_idx in range(n_locations):
                lat, lon = lats[loc_idx], lons[loc_idx]
                
                # Temperatura base com sazonalidade e localização
                temp_base = 25 - 0.6 * lat + 8 * np.sin(2 * np.pi * day_of_year / 365)
                temp_enso_effect = 2 * enso_index if lat < -10 else 1 * enso_index
                temperature = temp_base + temp_enso_effect + np.random.normal(0, 2)
                
                # Pressão atmosférica
                pressure_base = 1013 + 10 * np.sin(np.radians(lat)) 
                pressure_seasonal = 5 * np.cos(2 * np.pi * day_of_year / 365)
                pressure = pressure_base + pressure_seasonal + np.random.normal(0, 3)
                
                # Umidade relativa
                humidity_base = 70 + 10 * np.sin(np.radians(lat + 30))
                humidity_seasonal = 15 * np.sin(2 * np.pi * (day_of_year + 90) / 365)
                humidity = humidity_base + humidity_seasonal + np.random.normal(0, 5)
                humidity = np.clip(humidity, 30, 95)
                
                # Precipitação (correlacionada com umidade e ENSO)
                precip_prob = (humidity - 30) / 65 * 0.3 + 0.1 * enso_index
                precipitation = np.random.exponential(5) if np.random.random() < precip_prob else 0
                
                # Vento
                wind_speed = 3 + 2 * np.abs(nao_index) + np.random.exponential(2)
                wind_direction = np.random.uniform(0, 360)
                
                # Variáveis derivadas
                dewpoint = temperature - ((100 - humidity) / 5)
                heat_index = self.calculate_heat_index(temperature, humidity)
                
                # Padrões sinóticos simulados
                gradient_pressure = np.random.normal(0, 2)  # Gradiente de pressão
                vorticity = np.random.normal(0, 1)  # Vorticidade
                
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
                    'enso_index': enso_index,
                    'nao_index': nao_index,
                    'pressure_gradient': gradient_pressure,
                    'vorticity': vorticity,
                    # Targets (próximos 3 dias)
                    'temp_1d': temperature + np.random.normal(0, 1),
                    'temp_3d': temperature + np.random.normal(0, 2),
                    'pressure_1d': pressure + np.random.normal(0, 2),
                    'pressure_3d': pressure + np.random.normal(0, 3),
                    'precip_1d': max(0, precipitation + np.random.normal(0, 2)),
                    'precip_3d': max(0, precipitation + np.random.normal(0, 3))
                })
        
        return pd.DataFrame(data)
    
    def calculate_heat_index(self, temp, humidity):
        """
        Calcula índice de calor
        """
        if temp < 27:
            return temp
        
        hi = (0.5 * (temp + 61.0 + ((temp - 68.0) * 1.2) + (humidity * 0.094)))
        
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
        df_features['pressure_gradient_abs'] = np.abs(df_features['pressure_gradient'])
    
        # Verificar se location_id existe e se há múltiplas localizações
        has_location_id = 'location_id' in df_features.columns
        multiple_locations = has_location_id and len(df_features['location_id'].unique()) > 1
    
        # Features de lag (valores anteriores)
        lag_columns = ['temperature', 'pressure', 'humidity']
        for col in lag_columns:
            if col in df_features.columns:
                if multiple_locations:
                    # Múltiplas localizações - usar groupby
                    df_features[f'{col}_lag1'] = df_features.groupby('location_id')[col].shift(1)
                    df_features[f'{col}_lag3'] = df_features.groupby('location_id')[col].shift(3)
                else:
                    # Uma localização ou sem location_id - shift simples
                    df_features[f'{col}_lag1'] = df_features[col].shift(1)
                    df_features[f'{col}_lag3'] = df_features[col].shift(3)
    
        # Features de tendência
        trend_columns = ['temperature', 'pressure']
        for col in trend_columns:
            if col in df_features.columns:
                if multiple_locations:
                    # Múltiplas localizações - usar groupby
                    df_features[f'{col}_trend'] = (df_features.groupby('location_id')[col].diff() / 
                                            df_features.groupby('location_id')[col].shift(1) * 100)
                else:
                    # Uma localização ou sem location_id - diff simples
                    df_features[f'{col}_trend'] = (df_features[col].diff() / 
                                            df_features[col].shift(1) * 100)
    
        # Preencher NaN com médias
        df_features = df_features.fillna(df_features.mean())
    
        return df_features
    
    def prepare_lstm_data(self, df, sequence_length=7, target_col='temp_1d'):
        """
        Prepara dados para modelo LSTM
        """
        sequences = []
        targets = []
        
        feature_cols = ['temperature', 'pressure', 'humidity', 'wind_speed', 
                       'enso_index', 'nao_index', 'sin_day', 'cos_day']
        
        for location in df['location_id'].unique():
            location_data = df[df['location_id'] == location].sort_values('date')
            
            for i in range(len(location_data) - sequence_length):
                seq = location_data[feature_cols].iloc[i:i+sequence_length].values
                target = location_data[target_col].iloc[i+sequence_length]
                
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def train_ensemble_models(self, df, target_variables=None):
        """
        Treina conjunto de modelos de ML
        """
        if target_variables is None:
            target_variables = ['temp_1d', 'temp_3d', 'pressure_1d', 'pressure_3d', 
                              'precip_1d', 'precip_3d']
        
        # Preparar features
        df_features = self.create_features(df)
        
        # Selecionar features para treinamento
        feature_cols = [col for col in df_features.columns 
                       if col not in target_variables + ['date', 'location_id']]
        
        self.feature_names = feature_cols
        X = df_features[feature_cols]
        
        results = {}
        
        for target in target_variables:
            print(f"Treinando modelos para {target}...")
            
            y = df_features[target]
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Escalar dados
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[target] = scaler
            
            # Modelos
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
            
            target_results = {}
            
            for name, model in models.items():
                if name == 'MLP':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Métricas
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                target_results[name] = {
                    'model': model,
                    'mae': mae,
                    'mse': mse,
                    'r2': r2,
                    'predictions': y_pred,
                    'actual': y_test.values
                }
                
                print(f"  {name} - MAE: {mae:.3f}, R²: {r2:.3f}")
            
            # Ensemble (média ponderada baseada em R²)
            weights = {name: max(0, results['r2']) for name, results in target_results.items()}
            total_weight = sum(weights.values())
            
            if total_weight > 0:
                weights = {name: w/total_weight for name, w in weights.items()}
                
                ensemble_pred = sum(weights[name] * target_results[name]['predictions'] 
                                  for name in weights.keys())
                
                ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                ensemble_r2 = r2_score(y_test, ensemble_pred)
                
                target_results['Ensemble'] = {
                    'weights': weights,
                    'mae': ensemble_mae,
                    'r2': ensemble_r2,
                    'predictions': ensemble_pred,
                    'actual': y_test.values
                }
                
                print(f"  Ensemble - MAE: {ensemble_mae:.3f}, R²: {ensemble_r2:.3f}")
            
            results[target] = target_results
            self.models[target] = target_results
        
        return results
    
    def train_lstm_model(self, df, target_col='temp_1d', sequence_length=7):
        """
        Treina modelo LSTM para previsão de séries temporais
        """
        print(f"Treinando modelo LSTM para {target_col}...")
        
        # Preparar dados
        df_features = self.create_features(df)
        X, y = self.prepare_lstm_data(df_features, sequence_length, target_col)
        
        # Dividir dados
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Escalar dados
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        
        X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1]))
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Construir modelo LSTM
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
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
        
        print(f"LSTM - MAE: {mae:.3f}, R²: {r2:.3f}")
        
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
    
    def analyze_teleconnections(self, df):
        """
        Analisa correlações com índices de teleconexão
        """
        teleconnections = ['enso_index', 'nao_index']
        weather_vars = ['temperature', 'precipitation', 'pressure']
        
        correlations = {}
        
        for tele in teleconnections:
            correlations[tele] = {}
            for var in weather_vars:
                # Correlação geral
                corr = df[tele].corr(df[var])
                correlations[tele][var] = corr
                
                # Correlação por região (dividir em norte/sul)
                north = df[df['lat'] > -15]
                south = df[df['lat'] <= -15]
                
                correlations[tele][f'{var}_north'] = north[tele].corr(north[var]) if len(north) > 0 else np.nan
                correlations[tele][f'{var}_south'] = south[tele].corr(south[var]) if len(south) > 0 else np.nan
        
        return correlations
    
    def predict_weather(self, input_data, target_variable):
        """
        Faz previsão usando ensemble de modelos
        """
        if target_variable not in self.models:
            raise ValueError(f"Modelo para {target_variable} não foi treinado")
    
        # Preparar dados de entrada
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
    
        # Debug: verificar colunas de entrada
        print("Colunas do input_df antes de create_features:", input_df.columns.tolist())
    
        # Criar features
        input_features = self.create_features(input_df)
    
        # Debug: verificar features criadas
        print("Features criadas:", input_features.columns.tolist())
        print("Feature names esperadas:", self.feature_names)
    
        # Verificar se todas as features necessárias estão presentes
        missing_features = set(self.feature_names) - set(input_features.columns)
        if missing_features:
            print(f"Aviso: Features ausentes: {missing_features}")
            # Adicionar features ausentes com valor 0
            for feature in missing_features:
                input_features[feature] = 0
    
        # Selecionar apenas as features usadas no treinamento
        X = input_features[self.feature_names]
        # Preencher NaN com 0 para evitar erros nos modelos
        X = X.fillna(0)
    
        # Fazer previsões com todos os modelos
        models = self.models[target_variable]
        predictions = {}
    
        for name, model_info in models.items():
            if name == 'Ensemble':
                continue
            
            try:
                if name == 'MLP':
                    # Preencher NaN antes de escalar para MLP
                    scaler = self.scalers[target_variable]
                    X_no_nan = X.fillna(0)
                    X_scaled = scaler.transform(X_no_nan)
                    pred = model_info['model'].predict(X_scaled)
                else:
                    pred = model_info['model'].predict(X)
            
                predictions[name] = pred
            except Exception as e:
                print(f"Erro ao fazer previsão com {name}: {e}")
                predictions[name] = np.array([0])  # Valor padrão em caso de erro
    
        # Ensemble prediction
        if 'Ensemble' in models and len(predictions) > 0:
            weights = models['Ensemble']['weights']
            ensemble_pred = sum(weights.get(name, 0) * predictions[name] for name in predictions.keys())
            predictions['Ensemble'] = ensemble_pred
    
        return predictions
    
    def create_forecast_visualization(self, predictions_dict=None, save_path_prefix="forecast_fig"):
        """
        Cria visualização das previsões e salva cada gráfico como figura separada.
        Salva as imagens na pasta 'resultados_forecastSynopticML'.
        """
        output_dir = "resultados_forecastSynopticML"
        os.makedirs(output_dir, exist_ok=True)

        targets = ['temp_1d', 'temp_3d', 'pressure_1d', 'pressure_3d', 'precip_1d', 'precip_3d']
        titles = ['Temperatura (1 dia)', 'Temperatura (3 dias)', 
                  'Pressão (1 dia)', 'Pressão (3 dias)',
                  'Precipitação (1 dia)', 'Precipitação (3 dias)']

        for i, (target, title) in enumerate(zip(targets, titles)):
            if target in self.models:
                models_data = self.models[target]
                model_names = []
                maes = []
                r2s = []

                for name, data in models_data.items():
                    if 'mae' in data:
                        model_names.append(name)
                        maes.append(data['mae'])
                        r2s.append(data['r2'])

                fig, ax = plt.subplots(figsize=(7, 5))
                bars = ax.bar(model_names, maes, alpha=0.7)
                ax.set_title(f'{title} - MAE')
                ax.set_ylabel('Mean Absolute Error')
                ax.tick_params(axis='x', rotation=45)

                for j, (bar, r2) in enumerate(zip(bars, r2s)):
                    if r2 > 0.8:
                        bar.set_color('green')
                    elif r2 > 0.6:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')

                for j, (mae, r2) in enumerate(zip(maes, r2s)):
                    ax.text(j, mae + max(maes) * 0.01, f'R²: {r2:.3f}', 
                            ha='center', va='bottom', fontsize=8)

                plt.tight_layout()
                fig_filename = os.path.join(output_dir, f"{save_path_prefix}_{target}.png")
                plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    def create_prediction_plots(self, target_variable, save_path_prefix=None):
        """
        Cria gráficos de dispersão das previsões vs valores reais e salva cada gráfico como figura separada.
        Salva as imagens na pasta 'resultados_forecastSynopticML'.
        """
        output_dir = "resultados_forecastSynopticML"
        os.makedirs(output_dir, exist_ok=True)

        if target_variable not in self.models:
            print(f"Modelo para {target_variable} não encontrado")
            return

        models_data = self.models[target_variable]
        for name, data in models_data.items():
            if 'mae' in data:
                actual = data['actual']
                predicted = data['predictions']

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(actual, predicted, alpha=0.6)
                ax.plot([actual.min(), actual.max()],
                        [actual.min(), actual.max()], 'r--', lw=2)
                ax.set_xlabel('Valores Reais')
                ax.set_ylabel('Valores Preditos')
                ax.set_title(f'{name}\nMAE: {data["mae"]:.3f}, R²: {data["r2"]:.3f}')
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                if save_path_prefix:
                    fig_filename = os.path.join(output_dir, f"{save_path_prefix}_{target_variable}_{name}.png")
                else:
                    fig_filename = os.path.join(output_dir, f"prediction_{target_variable}_{name}.png")
                plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    def create_teleconnection_analysis(self, df, save_path_prefix="teleconnection_analysis"):
        """
        Cria visualização da análise de teleconexões e salva cada gráfico como figura separada.
        Salva as imagens na pasta 'resultados_forecastSynopticML'.
        """
        output_dir = "resultados_forecastSynopticML"
        os.makedirs(output_dir, exist_ok=True)

        correlations = self.analyze_teleconnections(df)
        
        # ENSO correlations
        enso_data = correlations['enso_index']
        vars_enso = list(enso_data.keys())
        corrs_enso = list(enso_data.values())
        
        fig_enso, ax_enso = plt.subplots(figsize=(8, 6))
        ax_enso.barh(vars_enso, corrs_enso)
        ax_enso.set_title('Correlações ENSO')
        ax_enso.set_xlabel('Correlação')
        ax_enso.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        fig_enso.savefig(os.path.join(output_dir, f"{save_path_prefix}_enso.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_enso)
        
        # NAO correlations
        nao_data = correlations['nao_index']
        vars_nao = list(nao_data.keys())
        corrs_nao = list(nao_data.values())
        
        fig_nao, ax_nao = plt.subplots(figsize=(8, 6))
        ax_nao.barh(vars_nao, corrs_nao)
        ax_nao.set_title('Correlações NAO')
        ax_nao.set_xlabel('Correlação')
        ax_nao.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        fig_nao.savefig(os.path.join(output_dir, f"{save_path_prefix}_nao.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_nao)
        
        # Time series plots
        sample_data = df.sample(min(1000, len(df))).sort_values('date')
        
        fig_enso_ts, ax_enso_ts = plt.subplots(figsize=(10, 5))
        ax_enso_ts.plot(sample_data['date'], sample_data['enso_index'], label='ENSO')
        ax_enso_ts.plot(sample_data['date'], sample_data['temperature']/10, label='Temp/10')
        ax_enso_ts.set_title('ENSO vs Temperatura')
        ax_enso_ts.legend()
        ax_enso_ts.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        fig_enso_ts.savefig(os.path.join(output_dir, f"{save_path_prefix}_enso_timeseries.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_enso_ts)
        
        fig_nao_ts, ax_nao_ts = plt.subplots(figsize=(10, 5))
        ax_nao_ts.plot(sample_data['date'], sample_data['nao_index'], label='NAO')
        ax_nao_ts.plot(sample_data['date'], sample_data['pressure']/1000, label='Press/1000')
        ax_nao_ts.set_title('NAO vs Pressão')
        ax_nao_ts.legend()
        ax_nao_ts.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        fig_nao_ts.savefig(os.path.join(output_dir, f"{save_path_prefix}_nao_timeseries.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_nao_ts)
    
    def create_geographic_forecast_map(self, df, target_date=None, save_path="geographic_forecast_map.png"):
        """
        Cria mapa geográfico com previsões e salva como figura.
        Salva a imagem na pasta 'resultados_forecastSynopticML'.
        """
        output_dir = "resultados_forecastSynopticML"
        os.makedirs(output_dir, exist_ok=True)
        save_path_full = os.path.join(output_dir, os.path.basename(save_path))

        try:
            fig = plt.figure(figsize=(15, 10))
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            # Configurar mapa
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS)
            ax.add_feature(cfeature.OCEAN, color='lightblue')
            ax.add_feature(cfeature.LAND, color='lightgray')
            
            # Focar no Brasil
            ax.set_extent([-75, -30, -35, 10], crs=ccrs.PlateCarree())
            
            # Dados das estações
            unique_locations = df.groupby('location_id').agg({
                'lat': 'first',
                'lon': 'first', 
                'temperature': 'mean',
                'precipitation': 'mean'
            }).reset_index()
            
            # Plotar temperatura
            scatter = ax.scatter(unique_locations['lon'], unique_locations['lat'], 
                               c=unique_locations['temperature'], cmap='RdYlBu_r',
                               s=100, alpha=0.7, transform=ccrs.PlateCarree())
            
            plt.colorbar(scatter, ax=ax, label='Temperatura (°C)')
            plt.title('Distribuição de Temperatura - Estações Meteorológicas')
            
            plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except ImportError:
            print("Cartopy não disponível. Criando gráfico alternativo...")
            
            # Gráfico alternativo sem cartopy
            unique_locations = df.groupby('location_id').agg({
                'lat': 'first',
                'lon': 'first', 
                'temperature': 'mean',
                'precipitation': 'mean'
            }).reset_index()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Temperatura
            scatter1 = ax1.scatter(unique_locations['lon'], unique_locations['lat'], 
                                 c=unique_locations['temperature'], cmap='RdYlBu_r',
                                 s=100, alpha=0.7)
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude') 
            ax1.set_title('Temperatura Média por Estação')
            plt.colorbar(scatter1, ax=ax1, label='Temperatura (°C)')
            
            # Precipitação
            scatter2 = ax2.scatter(unique_locations['lon'], unique_locations['lat'], 
                                 c=unique_locations['precipitation'], cmap='Blues',
                                 s=100, alpha=0.7)
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
            ax2.set_title('Precipitação Média por Estação')
            plt.colorbar(scatter2, ax=ax2, label='Precipitação (mm)')
            
            plt.tight_layout()
            plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    def save_model(self, filename):
        """
        Salva o modelo treinado
        """
        model_data = {
            'models': {},
            'scalers': {},
            'feature_names': self.feature_names,
            'teleconnection_indices': self.teleconnection_indices
        }
        # Save each scaler with joblib and store file path
        for target, scaler in self.scalers.items():
            scaler_file = f"{filename}_scaler_{target}.joblib"
            joblib.dump(scaler, scaler_file)
            model_data['scalers'][target] = scaler_file
        
        # Salvar apenas dados serializáveis
        for target, models in self.models.items():
            model_data['models'][target] = {}
            for name, model_info in models.items():
                if name != 'Ensemble' and 'lstm' not in target:
                    # Salvar modelos sklearn
                    model_data['models'][target][name] = {
                        'mae': model_info['mae'],
                        'r2': model_info['r2'],
                        'model_file': f"{filename}_{target}_{name}.joblib"
                    }
                    joblib.dump(model_info['model'], f"{filename}_{target}_{name}.joblib")
                elif name == 'Ensemble':
                    model_data['models'][target][name] = {
                        'weights': model_info['weights'],
                        'mae': model_info['mae'],
                        'r2': model_info['r2']
                    }
        
        # Salvar metadados
        with open(f"{filename}_metadata.json", 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Modelo salvo como {filename}_metadata.json")
    
    def load_model(self, filename):
        """
        Carrega modelo salvo
        """
        # Carregar metadados
        with open(f"{filename}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        self.scalers = model_data['scalers']
        self.feature_names = model_data['feature_names']
        self.teleconnection_indices = model_data['teleconnection_indices']
        
        # Carregar modelos
        self.models = {}
        for target, models in model_data['models'].items():
            self.models[target] = {}
            for name, model_info in models.items():
                if name != 'Ensemble':
                    model_info['model'] = joblib.load(model_info['model_file'])
                self.models[target][name] = model_info
        
        print(f"Modelo carregado de {filename}_metadata.json")
    
    def generate_forecast_report(self, df, save_path=None):
        """
        Gera relatório completo de previsão
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_samples': len(df),
                'date_range': f"{df['date'].min()} to {df['date'].max()}",
                'locations': df['location_id'].nunique(),
                'variables': list(df.select_dtypes(include=[np.number]).columns)
            },
            'model_performance': {},
            'teleconnection_analysis': self.analyze_teleconnections(df)
        }
        
        # Performance dos modelos
        for target, models in self.models.items():
            report['model_performance'][target] = {}
            # Se models for um dicionário (ensemble/sklearn), iterar normalmente
            if isinstance(models, dict):
                for name, model_info in models.items():
                    if isinstance(model_info, dict) and 'mae' in model_info and 'r2' in model_info:
                        mae = model_info['mae']
                        r2 = model_info['r2']
                        # Se for lista, pega o primeiro elemento
                        if isinstance(mae, list):
                            mae = mae[0] if len(mae) > 0 else None
                        if isinstance(r2, list):
                            r2 = r2[0] if len(r2) > 0 else None
                        report['model_performance'][target][name] = {
                            'mae': float(mae) if mae is not None else None,
                            'r2': float(r2) if r2 is not None else None
                        }
            # Se models for um resultado LSTM (dict com 'mae' e 'r2')
            elif isinstance(models, object) and hasattr(models, 'get'):
                if 'mae' in models and 'r2' in models:
                    report['model_performance'][target] = {
                        'mae': float(models['mae']),
                        'r2': float(models['r2'])
                    }
        
        # Salvar relatório
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report


# Exemplo de uso da plataforma
def main():
    """
    Exemplo de uso completo da plataforma
    """
    print("=== Plataforma de Previsão Sinótica com Machine Learning ===")
    
    # Inicializar sistema
    forecast_system = SynopticMLForecast()
    
    # Gerar dados sintéticos
    print("\n1. Gerando dados meteorológicos sintéticos...")
    df = forecast_system.generate_synthetic_weather_data(n_samples=3000, n_locations=15)
    print(f"Dataset criado com {len(df)} registros")
    print(f"Variáveis: {list(df.columns)}")
    
    # Análise exploratória
    print("\n2. Análise exploratória dos dados...")
    print("Estatísticas descritivas:")
    print(df[['temperature', 'pressure', 'humidity', 'precipitation']].describe())
    
    # Treinar modelos ensemble
    print("\n3. Treinando modelos ensemble...")
    results = forecast_system.train_ensemble_models(df)
    
    # Treinar modelo LSTM
    print("\n4. Treinando modelo LSTM...")
    lstm_results = forecast_system.train_lstm_model(df, target_col='temp_1d')
    
    # Análise de teleconexões
    print("\n5. Analisando teleconexões...")
    teleconnections = forecast_system.analyze_teleconnections(df)
    print("Correlações ENSO:")
    for var, corr in teleconnections['enso_index'].items():
        print(f"  {var}: {corr:.3f}")
    
    # Fazer previsão de exemplo
    print("\n6. Fazendo previsão de exemplo...")
    sample_input = {
        'lat': -15.7801,
        'lon': -47.9292, 
        'day_of_year': 180,
        'month': 6,
        'season': 2,
        'temperature': 22.5,
        'pressure': 1015.2,
        'humidity': 65.0,
        'precipitation': 0.0,
        'wind_speed': 5.2,
        'wind_direction': 180.0,
        'dewpoint': 15.2,
        'heat_index': 22.5,
        'enso_index': 0.5,
        'nao_index': -0.2,
        'pressure_gradient': 1.2,
        'vorticity': 0.1
    }
    
    predictions = forecast_system.predict_weather(sample_input, 'temp_1d')
    print("Previsões de temperatura para 1 dia:")
    for model, pred in predictions.items():
        print(f"  {model}: {pred[0]:.2f}°C")
    
    # Criar visualizações
    print("\n7. Criando visualizações...")
    forecast_system.create_forecast_visualization()
    forecast_system.create_prediction_plots('temp_1d')
    forecast_system.create_teleconnection_analysis(df)
    forecast_system.create_geographic_forecast_map(df)

    # Gerar relatório
    print("\n8. Gerando relatório...")
    report = forecast_system.generate_forecast_report(df, 'forecast_report.json')
    
    # Salvar modelo
    print("\n9. Salvando modelo...")
    forecast_system.save_model('synoptic_forecast_model')
    
    print("\n=== Análise completa finalizada! ===")
    print("Arquivos gerados:")
    print("- forecast_report.json: Relatório completo")
    print("- synoptic_forecast_model_metadata.json: Metadados do modelo")
    print("- synoptic_forecast_model_*.joblib: Modelos treinados")
    
    return forecast_system, df, results


# Classe para interface web simples
class ForecastWebInterface:
    """
    Interface web simples para a plataforma de previsão
    """
    def __init__(self, forecast_system):
        self.forecast_system = forecast_system

    def run(self, host="127.0.0.1", port=8080):
        """
        Inicia um servidor web local para a interface de previsão.
        """

        app = Flask(__name__)

        @app.route("/", methods=["GET", "POST"])
        def index():
            prediction = None
            confidence = None
            error = None
            if request.method == "POST":
                try:
                    # Coletar dados do formulário
                    input_data = {
                        "lat": float(request.form.get("lat", -15.7801)),
                        "lon": float(request.form.get("lon", -47.9292)),
                        "temperature": float(request.form.get("temperature", 25.0)),
                        "pressure": float(request.form.get("pressure", 1013.2)),
                        "humidity": float(request.form.get("humidity", 70.0)),
                        "wind_speed": float(request.form.get("wind_speed", 5.0)),
                        "enso_index": float(request.form.get("enso_index", 0.0)),
                        "nao_index": float(request.form.get("nao_index", 0.0)),
                        # Preencher valores padrão para variáveis obrigatórias
                        "day_of_year": 180,
                        "month": 6,
                        "season": 2,
                        "precipitation": 0.0,
                        "wind_direction": 180.0,
                        "dewpoint": 15.0,
                        "heat_index": 25.0,
                        "pressure_gradient": 1.0,
                        "vorticity": 0.0
                    }
                    preds = self.forecast_system.predict_weather(input_data, "temp_1d")
                    if "Ensemble" in preds:
                        prediction = float(preds["Ensemble"][0])
                        confidence = "R²: {:.1f}%".format(
                            100 * self.forecast_system.models["temp_1d"]["Ensemble"]["r2"]
                        )
                    else:
                        # Pega o primeiro modelo disponível
                        key = next(iter(preds))
                        prediction = float(preds[key][0])
                        confidence = "Modelo: {}".format(key)
                except Exception as ex:
                    error = f"Erro ao processar previsão: {ex}"

            html_form = self.create_input_form(prediction, confidence, error)
            # Salvar HTML na pasta resultados_forecastSynopticML
            output_dir = "resultados_forecastSynopticML"
            os.makedirs(output_dir, exist_ok=True)
            html_path = os.path.join(output_dir, "web_interface_last.html")
            try:
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_form)
            except Exception as e:
                print(f"Erro ao salvar HTML: {e}")
            return render_template_string(html_form)

        print(f"Interface web disponível em http://{host}:{port}")
        app.run(host=host, port=port, debug=False)

    def create_input_form(self, prediction=None, confidence=None, error=None):
        """
        Cria formulário HTML para entrada de dados e exibe resultado se houver.
        """
        html_form = f"""
        <html>
        <head>
            <title>Previsão Meteorológica - ML</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .form-group {{ margin: 15px 0; }}
                label {{ display: inline-block; width: 200px; font-weight: bold; }}
                input {{ width: 200px; padding: 5px; }}
                button {{ background-color: #4CAF50; color: white; padding: 10px 20px; 
                        font-size: 16px; border: none; cursor: pointer; }}
                .result {{ background-color: #f0f8ff; padding: 20px; margin-top: 20px; 
                         border-radius: 5px; }}
                .error {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Sistema de Previsão Meteorológica com Machine Learning</h1>
            
            <form id="forecastForm" method="post">
                <h3>Dados da Estação Meteorológica</h3>
                
                <div class="form-group">
                    <label>Latitude:</label>
                    <input type="number" step="0.0001" name="lat" value="-15.7801" required>
                </div>
                
                <div class="form-group">
                    <label>Longitude:</label>
                    <input type="number" step="0.0001" name="lon" value="-47.9292" required>
                </div>
                
                <div class="form-group">
                    <label>Temperatura (°C):</label>
                    <input type="number" step="0.1" name="temperature" value="25.0" required>
                </div>
                
                <div class="form-group">
                    <label>Pressão (hPa):</label>
                    <input type="number" step="0.1" name="pressure" value="1013.2" required>
                </div>
                
                <div class="form-group">
                    <label>Umidade (%):</label>
                    <input type="number" step="0.1" name="humidity" value="70.0" required>
                </div>
                
                <div class="form-group">
                    <label>Velocidade do Vento (m/s):</label>
                    <input type="number" step="0.1" name="wind_speed" value="5.0" required>
                </div>
                
                <div class="form-group">
                    <label>Índice ENSO:</label>
                    <input type="number" step="0.1" name="enso_index" value="0.0" required>
                </div>
                
                <div class="form-group">
                    <label>Índice NAO:</label>
                    <input type="number" step="0.1" name="nao_index" value="0.0" required>
                </div>
                
                <button type="submit">Gerar Previsão</button>
            </form>
            
            {"<div class='result'><h3>Resultados da Previsão</h3>" if prediction is not None or error else ""}
            {f"<p><strong>Temperatura prevista (1 dia):</strong> {prediction:.2f}°C</p>" if prediction is not None else ""}
            {f"<p><strong>Confiança:</strong> {confidence}</p>" if confidence else ""}
            {f"<p class='error'>{error}</p>" if error else ""}
            {"</div>" if prediction is not None or error else ""}
        </body>
        </html>
        """
        return html_form


# Utilitários para processamento de dados reais
class WeatherDataProcessor:
    """
    Processador para dados meteorológicos reais
    """
    @staticmethod
    def load_csv_data(filepath):
        """
        Carrega dados de arquivo CSV
        """
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Erro ao carregar arquivo: {e}")
            return None
    
    @staticmethod
    def process_noaa_data(df):
        """
        Processa dados no formato NOAA
        """
        # Mapear colunas NOAA para formato padrão
        column_mapping = {
            'TEMP': 'temperature',
            'DEWP': 'dewpoint', 
            'SLP': 'pressure',
            'VISIB': 'visibility',
            'WDSP': 'wind_speed',
            'WDDIR': 'wind_direction',
            'PRCP': 'precipitation'
        }
        
        df_processed = df.rename(columns=column_mapping)
        
        # Converter unidades se necessário
        if 'temperature' in df_processed.columns:
            # Converter Fahrenheit para Celsius se necessário
            if df_processed['temperature'].max() > 50:
                df_processed['temperature'] = (df_processed['temperature'] - 32) * 5/9
        
        return df_processed
    
    @staticmethod
    def calculate_derived_variables(df):
        """
        Calcula variáveis meteorológicas derivadas
        """
        df_derived = df.copy()
        
        # Índice de calor
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df_derived['heat_index'] = df.apply(
                lambda row: SynopticMLForecast().calculate_heat_index(
                    row['temperature'], row.get('humidity', 50)
                ), axis=1
            )
        
        # Ponto de orvalho a partir de temperatura e umidade
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df_derived['dewpoint_calc'] = df_derived['temperature'] - (
                (100 - df_derived['humidity']) / 5
            )
        
        # Velocidade do vento em diferentes unidades
        if 'wind_speed' in df.columns:
            df_derived['wind_speed_kmh'] = df_derived['wind_speed'] * 3.6
            df_derived['wind_speed_knots'] = df_derived['wind_speed'] * 1.944
        
        return df_derived


if __name__ == "__main__":
    # Executar exemplo principal
    forecast_system, data, results = main()