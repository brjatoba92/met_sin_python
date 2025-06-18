# Dependencias do Projeto
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

class SynopticMeteorologyAnalyzer:
    def __init__(self, api_key=None):
        """
        Sistema para análise automática de cartas sinóticas
        
        Args:
            api_key: Chave da API OpenWeatherMap (opcional)
        """
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
    
    def get_weather_data(self, cities=None):
        """
        Coleta dados meteorológicos de múltiplas cidades
        """
        if cities is None:
            # Principais cidades do Brasil para análise sinótica
            cities = [
                {"name": "São Paulo", "lat": -23.5505, "lon": -46.6333},
                {"name": "Rio de Janeiro", "lat": -22.9068, "lon": -43.1729},
                {"name": "Brasília", "lat": -15.7942, "lon": -47.8822},
                {"name": "Salvador", "lat": -12.9714, "lon": -38.5014},
                {"name": "Fortaleza", "lat": -3.7172, "lon": -38.5433},
                {"name": "Manaus", "lat": -3.1190, "lon": -60.0217},
                {"name": "Porto Alegre", "lat": -30.0346, "lon": -51.2177},
                {"name": "Curitiba", "lat": -25.4244, "lon": -49.2654},
                {"name": "Recife", "lat": -8.0476, "lon": -34.8770},
                {"name": "Belém", "lat": -1.4558, "lon": -48.5044}
            ]
        
        weather_data = []
        
        for city in cities:
            if self.api_key:
                # Usando API real
                url = f"{self.base_url}/weather"
                params = {
                    'lat': city['lat'],
                    'lon': city['lon'],
                    'appid': self.api_key,
                    'units': 'metric'
                }
                
                try:
                    response = requests.get(url, params=params)
                    data = response.json()
                    
                    weather_data.append({
                        'city': city['name'],
                        'lat': city['lat'],
                        'lon': city['lon'],
                        'temperature': data['main']['temp'],
                        'pressure': data['main']['pressure'],
                        'humidity': data['main']['humidity'],
                        'wind_speed': data['wind']['speed'],
                        'wind_dir': data['wind'].get('deg', 0),
                        'datetime': datetime.now()
                    })
                except Exception as e:
                    print(f"Erro ao obter dados de {city['name']}: {e}")
            else:
                # Dados simulados para demonstração
                weather_data.append({
                    'city': city['name'],
                    'lat': city['lat'],
                    'lon': city['lon'],
                    'temperature': np.random.normal(25, 8),
                    'pressure': np.random.normal(1013, 15),
                    'humidity': np.random.normal(70, 20),
                    'wind_speed': np.random.normal(5, 3),
                    'wind_dir': np.random.uniform(0, 360),
                    'datetime': datetime.now()
                })
        
        return pd.DataFrame(weather_data)
    
    
    def interpolate_field(self, df, field, grid_size=50):
        """
        Interpola dados meteorológicos em uma grade regular
        """
        # limites de grade - Brasil
        lon_min, lon_max = -75, -30
        lat_min, lat_max = -35, 10

        # Criar grade regular
        lon_grid = np.linspace(lon_min, lon_max, grid_size)
        lat_grid = np.linspace(lat_min, lat_max, grid_size)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        # Interpolar dados
        points = df[['lon', 'lat']].values
        values = df[field].values

        interpolated = griddata(
            points, values, 
            (lon_mesh, lat_mesh), 
            method='linear',
            fill_value=np.nan
        )

        return lon_mesh, lat_mesh, interpolated
    
    def detect_pressure_centers(self, lon_mesh, lat_mesh, pressure_field):
        """
        Detecta centros de alta e baixa pressão
        """
        from scipy.ndimage import maximum_filter, minimum_filter

        # Filtros para encontrar máximos e mínimos locais
        max_filter = maximum_filter(pressure_field, size=5)
        min_filter = minimum_filter(pressure_field, size=5)

        # Identificar centros de alta pressão
        high_centers = []
        highs = (pressure_field == max_filter) & (pressure_field > np.nanmean(pressure_field) + np.nanstd(pressure_field))
        high_indices = np.where(highs)

        for i, j in zip(high_indices[0], high_indices[1]):
            if not np.isnan(pressure_field[i, j]):
                high_centers.append({
                    'type': 'HIGH',
                    'lat': lat_mesh[i, j],
                    'lon': lon_mesh[i, j],
                    'pressure': pressure_field[i, j]
                })
        
        # Identificar centros de baixa pressão
        low_centers = []
        lows = (pressure_field == min_filter) & (pressure_field < np.nanmean(pressure_field) - np.nanstd(pressure_field))
        low_indices = np.where(lows)

        for i, j in zip(low_indices[0], low_indices[1]):
            if not np.isnan(pressure_field[i, j]):
                low_centers.append({
                    'type': 'LOW',
                    'lat': lat_mesh[i, j],
                    'lon': lon_mesh[i, j],
                    'pressure': pressure_field[i, j]
                })
        
        return high_centers + low_centers