# Importação de dependencias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta
import json
import warnings

warnings.filterwarnings("ignore")

class WeatherSystemTracker:
    def __init__(self):
        """
        Sistema para detecção e rastreamento de sistemas meteorológicos
        """
        self.systems_history = []
        self.fronts_history = []
        self.tracking_threshold = 300 # km máximo para associar sistemas
    
    def generate_synthetic_pressure_field(self, time_step=0):
        """
        Gera um campo sintético de pressão para um dado passo de tempo
        """
        # Grade de coordenadas (Brasil)
        lon = np.linspace(-75, -30, 100) # Cria 100 valores igualmente espaçados de -75 até -30 (longitude)
        lat = np.linspace(-35, 10, 80)
        LON, LAT = np.meshgrid(lon, lat) # Cria uma grade 2D com as coordenadas

        # Campo sintético de pressão (exemplo)
        pressure = 1013 + 5 * np.sin(np.radians(LAT * 2))

        # Sistema de baixa pressão (ciclone) em movimento
        cyclone_center_lon = -50 + 2 * time_step
        cyclone_center_lat = -20 + 0.5 * time_step
        cyclone_distance = np.sqrt((LON - cyclone_center_lon) ** 2 + (LAT - cyclone_center_lat) ** 2)
        cyclone_intensity = 25 * np.exp(-cyclone_distance ** 2 / 50)
        pressure -= cyclone_intensity

        # Sistema de alta pressão (anticyclone) em movimento
        anticyclone_center_lon = -40 - 1.5 * time_step
        anticyclone_center_lat = -25 + 0.3 * time_step
        anticyclone_distance = np.sqrt((LON - anticyclone_center_lon) ** 2 + (LAT - anticyclone_center_lat) ** 2)
        anticyclone_intensity = 20 * np.exp(-anticyclone_distance ** 2 / 60)
        pressure += anticyclone_intensity

        # Adicionar ruído realistico
        noise = np.random.normal(0, 2, pressure.shape) # Distribuição normal com media 0 e desvio padrão 2
        pressure += noise

        return LON, LAT, pressure
    
    def generate_synthetic_temperature_field(self, time_step=0):
        """
        Gera campo de temperatura sintético
        """
        lon = np.linspace(-75, -30, 100)
        lat = np.linspace(-35, 10, 80)
        LON, LAT = np.meshgrid(lon, lat)

        # Gradiente latitudinal de temperatura
        temperature = 25 - 0.7 * LAT

        # Frente fria em movimento
        front_position = -60 + 3 * time_step
        front_gradient = 10 / (1 + np.exp((LON - front_position) * 0.3))
        temperature -= front_gradient

        # Variabilidade espacial
        temp_variation = 3 *  np.sin(np.radians(LON * 2) * np.cos(np.radians(LAT * 1.5)))
        temperature += temp_variation

        # Ruido
        noise = np.random.normal(0, 1, temperature.shape)
        temperature += noise

        return LON, LAT, temperature
    
    def detect_pressure_systems(self, lon_mesh, lat_mesh, pressure_field, min_intensity=5):
        """
        Detecta sistemas de pressão usando análise de extremos locais
        """
        # Aplicar filtro gaussiano para suavizar
        smoothed_pressure = ndimage.gaussian_filter(pressure_field, sigma=1.5)

        # Detectar máximos locais (alta pressão)
        local_maxima = ndimage.maximum_filter(smoothed_pressure, size=5) == smoothed_pressure
        high_pressure_mask = smoothed_pressure > (np.nanmean(smoothed_pressure) + min_intensity)
        highs = local_maxima & high_pressure_mask

        # Detectar minimos locais (baixa pressão)
        local_minima = ndimage.minimum_filter(smoothed_pressure, size=5) == smoothed_pressure
        low_pressure_mask = smoothed_pressure < (np.nanmean(smoothed_pressure) - min_intensity)
        lows = local_minima & low_pressure_mask

        systems = []

        # Processar sistemas de alta pressão
        high_indices = np.where(highs)
        for i, j in zip(high_indices[0], high_indices[1]):
            if not np.isnan(pressure_field[i, j]):
                systems.append({
                    'type': 'HIGH',
                    'lat': lat_mesh[i, j],
                    'lon': lon_mesh[i, j],
                    'pressure': pressure_field[i, j],
                    'intensity': pressure_field[i, j] - np.nanmean(pressure_field),
                    'timestamp': datetime.now()

                })
        # Processar sistemas de baixa pressão
        low_indices = np.where(lows)
        for i, j in zip(low_indices[0], low_indices[1]):
            if not np.isnan(pressure_field[i, j]):
                systems.append({
                    'type': 'LOW',
                    'lat': lat_mesh[i, j],
                    'lon': lon_mesh[i, j],
                    'pressure': pressure_field[i, j],
                    'intensity': pressure_field[i, j] - np.nanmean(pressure_field),
                    'timestamp': datetime.now()
                })

        return systems

        