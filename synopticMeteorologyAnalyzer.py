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
from scipy.ndimage import maximum_filter, minimum_filter
import os

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
                {"name": "Belém", "lat": -1.4558, "lon": -48.5044},
                {"name": "Belo Horizonte", "lat": -19.9167, "lon": -43.9333},
                {"name": "Boa Vista", "lat": 2.8236, "lon": -60.6753},
                {"name": "Florianópolis", "lat": -27.5945, "lon": -48.5477},
                {"name": "Porto Velho", "lat": -8.7619, "lon": -63.9037},
                {"name": "João Pessoa", "lat": -7.1250, "lon": -34.8567},
                {"name": "Teresina", "lat": -5.0892, "lon": -42.7700},
                {"name": "Aracaju", "lat": -10.9472, "lon": -37.0731},
                {"name": "Campo Grande", "lat": -20.4697, "lon": -54.6201},
                {"name": "Cuiabá", "lat": -15.6014, "lon": -56.0979},
                {"name": "Goiânia", "lat": -16.6869, "lon": -49.2648},
                {"name": "Macapá", "lat": 0.0349, "lon": -51.0694},
                {"name": "Maceió", "lat": -9.6658, "lon": -35.7350},
                {"name": "Natal", "lat": -5.7945, "lon": -35.2110},
                {"name": "Palmas", "lat": -10.1840, "lon": -48.3336},
                {"name": "Rio Branco", "lat": -9.9747, "lon": -67.8243},
                {"name": "São Luís", "lat": -2.5307, "lon": -44.3068},
                {"name": "Vitória", "lat": -20.3155, "lon": -40.3128},
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
    
    def create_synoptic_chart(self, df, save_path=None):
        """
        Cria carta sinotica com analise automática
        """

        fig = plt.figure(figsize=(15, 12))

        # Configurar projeção
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([-75, -30, -35, 10], crs=ccrs.PlateCarree())

        # Adicionar caracteristicas geograficas
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.STATES, alpha=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)

        # Interpolar campo de pressão
        lon_mesh, lat_mesh, pressure_field = self.interpolate_field(df, 'pressure')

        # Plotar isolinhas de pressão
        pressure_contours = ax.contour(
            lon_mesh, lat_mesh, pressure_field,
            levels=np.arange(980, 1040, 4),
            colors='black',
            linewidths=1,
            transform=ccrs.PlateCarree()
        )
        ax.clabel(pressure_contours, inline=True, fontsize=8, fmt='%d')

        # Detectar e plotar centros de pressão
        centers = self.detect_pressure_centers(lon_mesh, lat_mesh, pressure_field)

        for center in centers:
            if center['type'] == 'HIGH':
                ax.plot(center['lon'], center['lat'], 'ro', markersize=12, 
                        transform=ccrs.PlateCarree())
                ax.text(center['lon'], center['lat'], 'A',
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        transform=ccrs.PlateCarree())
            else:
                ax.plot(center['lon'], center['lat'], 'bo', markersize=12, 
                        transform=ccrs.PlateCarree())
                ax.text(center['lon'], center['lat'], 'B',
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        transform=ccrs.PlateCarree())
        
        # Plotar estações meteorológicas
        for _, row in df.iterrows():
            ax.plot(row['lon'], row['lat'], 'ko', markersize=8, 
                    transform=ccrs.PlateCarree())
            
            # Plotar direção do vento
            if row['wind_speed'] > 1:
                dx = 0.5 * np.sin(np.radians(row['wind_dir']))
                dy = 0.5 * np.cos(np.radians(row['wind_dir']))
                ax.arrow(row['lon'], row['lat'], dx, dy,
                        head_width=0.2, head_length=0.2, fc='red', ec='red',
                        transform=ccrs.PlateCarree())
        
        lon_mesh_temp, lat_mesh_temp, temp_field = self.interpolate_field(df, 'temperature')
        # Plotar isolinhas de temperatura
        temp_contours = ax.contourf(
            lon_mesh_temp, lat_mesh_temp, temp_field,
            levels=np.arange(0, 40, 2),
            alpha=0.3,
            cmap='RdYlBu_r',
            transform=ccrs.PlateCarree()
        )

        # Adicionar barra de cores para temperatura
        cbar = plt.colorbar(temp_contours, ax=ax, shrink=0.7, pad=0.1)
        cbar.set_label('Temperatura (°C)', fontsize=12)

        # Configurações do gráfico
        ax.gridlines(draw_labels=True, alpha=0.5)
        plt.title(f'Carta Sinótica - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=16, fontweight='bold')

        # Adicionar legenda
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Alta Pressão', markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='Baixa Pressão', markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='Estação Meteorológica', markersize=8),
            plt.Line2D([0], [0], marker='o', color='red', label='Direção do Vento', markersize=8)
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Carta sinótica salva em: {save_path}")
        
        return fig, centers
        
    def generate_weather_report(self, df, centers):
        """
        Gera relatório meteorológico automático
        """
        report = []
        report.append(f"RELAÓRIO METEORÓLOGICO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)

        # Estatisticas gerais
        report.append("\nCONDIÇÕES GERAIS: ")
        report.append(f"Temperatura Média: {df['temperature'].mean():.1f} °C")
        report.append(f"Pressão Média: {df['pressure'].mean():.1f} hPa")
        report.append(f"Umidade Média: {df['humidity'].mean():.1f} %")
        report.append(f"Velocidade do Vento Média: {df['wind_speed'].mean():.1f} m/s")

        # Análise de sistemas de pressão
        if centers:
            report.append(f"\nSISTEMAS DE PRESSÃO IDENTIFICADOS: {len(centers)}")
            for i, center in enumerate(centers, 1):
                system_type = "ALTA PRESSÃO" if center['type'] == 'HIGH' else "BAIXA PRESSÃO"
                report.append(f"{i}. {system_type}")
                report.append(f" Localização: {center['lat']:.2f}°S, {abs(center['lon']):.2f}°W")
                report.append(f" Pressão central: {center['pressure']:.1f} hPa")
        
        # Condições por região
        report.append("\nCONDICOES POR REGIAO: ")
        for _, row in df.iterrows():
            report.append(f"\n{row['city']}:")
            report.append(f" Temperatura: {row['temperature']:.1f} °C")
            report.append(f" Pressão: {row['pressure']:.1f} hPa")
            report.append(f" Umidade: {row['humidity']:.1f} %")
            report.append(f" Vento: {row['wind_speed']:.1f} m/s, {row['wind_dir']:.1f}°")

        return "\n".join(report)

if __name__ == "__main__":
     # Inicializar analisador (sem API key usará dados simulados)
    analyzer = SynopticMeteorologyAnalyzer() 
    #analyzer = SynopticMeteorologyAnalyzer(api_key='SUA_CHAVE_API')
    
    # Coletar dados meteorologicos
    weather_df = analyzer.get_weather_data()
    print(f"Dados coletados de {len(weather_df)} estações meteorológicas.")

    # Criar pasta 'resultados' se não existir
    resultados_dir = "resultados"
    os.makedirs(resultados_dir, exist_ok=True)

    # Criar carta sinótica
    print("Gerando carta sinótica...")
    fig, pressure_centers = analyzer.create_synoptic_chart(
        weather_df, 
        save_path=os.path.join(resultados_dir, 'carta_sinotica.png')
    )

    # Gerar relatório meteorológico
    print("Gerando relatório meteorológico...")
    report = analyzer.generate_weather_report(weather_df, pressure_centers)
    print("\n" + report)

    # Salvar relatório
    with open(os.path.join(resultados_dir, "relatorio_meteorologico.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    
    plt.show()