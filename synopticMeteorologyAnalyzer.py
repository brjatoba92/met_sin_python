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