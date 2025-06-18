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
        noise = np.random.normal(0, 2, pressure.shape)
        pressure += noise

        return LON, LAT, pressure