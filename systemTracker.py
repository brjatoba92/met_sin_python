# Detector e Rastreador de Sistemas Meteorológicos
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
import os
warnings.filterwarnings('ignore')

class WeatherSystemTracker:
    def __init__(self):
        """
        Sistema para detecção e rastreamento de sistemas meteorológicos
        """
        self.systems_history = []
        self.fronts_history = []
        self.tracking_threshold = 300  # km máximo para associar sistemas
        
    def generate_synthetic_pressure_field(self, time_step=0):
        """
        Gera campo de pressão sintético com sistemas meteorológicos em movimento
        """
        # Grade de coordenadas (Brasil)
        lon = np.linspace(-75, -30, 100)
        lat = np.linspace(-35, 10, 80)
        LON, LAT = np.meshgrid(lon, lat)
        
        # Campo base de pressão
        pressure = 1013 + 5 * np.sin(np.radians(LAT * 2))
        
        # Sistema de baixa pressão (ciclone) em movimento
        cyclone_center_lon = -50 + 2 * time_step
        cyclone_center_lat = -20 + 0.5 * time_step
        cyclone_distance = np.sqrt((LON - cyclone_center_lon)**2 + (LAT - cyclone_center_lat)**2)
        cyclone_intensity = 25 * np.exp(-cyclone_distance**2 / 50)
        pressure -= cyclone_intensity
        
        # Sistema de alta pressão (anticiclone)
        anticyclone_center_lon = -40 - 1.5 * time_step
        anticyclone_center_lat = -25 + 0.3 * time_step
        anticyclone_distance = np.sqrt((LON - anticyclone_center_lon)**2 + (LAT - anticyclone_center_lat)**2)
        anticyclone_intensity = 20 * np.exp(-anticyclone_distance**2 / 60)
        pressure += anticyclone_intensity
        
        # Adicionar ruído realístico
        noise = np.random.normal(0, 2, pressure.shape)
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
        temperature = 30 - 0.7 * LAT
        
        # Frente fria em movimento
        front_position = -60 + 3 * time_step
        front_gradient = 10 / (1 + np.exp(-(LON - front_position) * 0.3))
        temperature -= front_gradient
        
        # Variabilidade espacial
        temp_variation = 3 * np.sin(np.radians(LON * 2)) * np.cos(np.radians(LAT * 1.5))
        temperature += temp_variation
        
        # Ruído
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
        
        # Detectar mínimos locais (baixa pressão)
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
                    'intensity': np.nanmean(pressure_field) - pressure_field[i, j],
                    'timestamp': datetime.now()
                })
        
        return systems
    
    def detect_fronts(self, lon_mesh, lat_mesh, temperature_field, threshold=2.0):
        """
        Detecta frentes meteorológicas usando gradientes de temperatura
        """
        # Calcular gradientes de temperatura
        grad_y, grad_x = np.gradient(temperature_field)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Encontrar regiões com gradientes altos
        front_mask = grad_magnitude > threshold
        
        # Usar clustering para agrupar pontos de frente
        if np.any(front_mask):
            front_points = np.column_stack(np.where(front_mask))
            
            if len(front_points) > 5:
                # Aplicar DBSCAN para agrupar pontos próximos
                clustering = DBSCAN(eps=3, min_samples=5).fit(front_points)
                
                fronts = []
                for cluster_id in set(clustering.labels_):
                    if cluster_id != -1:  # Ignorar ruído
                        cluster_points = front_points[clustering.labels_ == cluster_id]
                        
                        # Calcular centroide da frente
                        center_i = int(np.mean(cluster_points[:, 0]))
                        center_j = int(np.mean(cluster_points[:, 1]))
                        
                        # Determinar tipo de frente baseado no gradiente
                        local_grad_x = grad_x[center_i, center_j]
                        local_grad_y = grad_y[center_i, center_j]
                        
                        front_type = 'COLD' if local_grad_x < 0 else 'WARM'
                        
                        fronts.append({
                            'type': front_type,
                            'lat': lat_mesh[center_i, center_j],
                            'lon': lon_mesh[center_i, center_j],
                            'intensity': grad_magnitude[center_i, center_j],
                            'direction': np.degrees(np.arctan2(local_grad_y, local_grad_x)),
                            'points': [(lat_mesh[i, j], lon_mesh[i, j]) for i, j in cluster_points],
                            'timestamp': datetime.now()
                        })
                
                return fronts
        
        return []
    
    def track_systems(self, new_systems, time_step):
        """
        Rastreia sistemas meteorológicos ao longo do tempo
        """
        if not self.systems_history or not new_systems:
            # Adicionar IDs únicos aos novos sistemas
            for i, system in enumerate(new_systems):
                system['id'] = f"{system['type']}_{time_step}_{i}"
                system['track'] = [(system['lat'], system['lon'])]
            
            self.systems_history.extend(new_systems)
            return new_systems
        
        # Última detecção
        last_systems = [s for s in self.systems_history if 'track' in s]
        
        tracked_systems = []
        
        for new_system in new_systems:
            best_match = None
            min_distance = float('inf')
            
            # Encontrar o sistema mais próximo do mesmo tipo
            for old_system in last_systems:
                if old_system['type'] == new_system['type']:
                    # Calcular distância (aproximação simples)
                    distance = np.sqrt(
                        (new_system['lat'] - old_system['lat'])**2 + 
                        (new_system['lon'] - old_system['lon'])**2
                    ) * 111  # Conversão aproximada para km
                    
                    if distance < self.tracking_threshold and distance < min_distance:
                        min_distance = distance
                        best_match = old_system
            
            if best_match:
                # Sistema rastreado - atualizar trajetória
                new_system['id'] = best_match['id']
                new_system['track'] = best_match['track'] + [(new_system['lat'], new_system['lon'])]
                new_system['speed'] = min_distance  # km entre detecções
            else:
                # Novo sistema
                new_system['id'] = f"{new_system['type']}_{time_step}_{len(tracked_systems)}"
                new_system['track'] = [(new_system['lat'], new_system['lon'])]
                new_system['speed'] = 0
            
            tracked_systems.append(new_system)
        
        self.systems_history.extend(tracked_systems)
        return tracked_systems
    
    def predict_trajectory(self, system, hours_ahead=24):
        """
        Prediz trajetória de um sistema usando regressão linear simples
        """
        if len(system['track']) < 2:
            return None
        
        track = np.array(system['track'])
        
        # Calcular velocidade média
        if len(track) >= 2:
            dlat = np.diff(track[:, 0])
            dlon = np.diff(track[:, 1])
            
            # Velocidade média por time step
            avg_dlat = np.mean(dlat)
            avg_dlon = np.mean(dlon)
            
            # Projetar posição futura
            current_lat, current_lon = track[-1]
            future_positions = []
            
            for h in range(1, hours_ahead + 1):
                future_lat = current_lat + avg_dlat * h
                future_lon = current_lon + avg_dlon * h
                future_positions.append((future_lat, future_lon))
            
            return future_positions
        
        return None
    
    def generate_alerts(self, systems):
        """
        Gera alertas meteorológicos baseados na intensidade dos sistemas
        """
        alerts = []
        
        for system in systems:
            alert_level = None
            
            if system['type'] == 'LOW':
                if system['intensity'] > 20:
                    alert_level = 'SEVERE'
                elif system['intensity'] > 10:
                    alert_level = 'MODERATE'
                elif system['intensity'] > 5:
                    alert_level = 'WATCH'
            
            elif system['type'] == 'HIGH':
                if system['intensity'] > 15:
                    alert_level = 'WATCH'
            
            if alert_level:
                alerts.append({
                    'system_id': system['id'],
                    'type': system['type'],
                    'level': alert_level,
                    'location': f"{system['lat']:.2f}°S, {abs(system['lon']):.2f}°W",
                    'intensity': system['intensity'],
                    'timestamp': system['timestamp']
                })
        
        return alerts
    
    def visualize_tracking(self, lon_mesh, lat_mesh, pressure_field, systems, save_path=None):
        """
        Visualiza sistemas detectados e suas trajetórias
        """

        fig = plt.figure(figsize=(16, 12))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([-75, -30, -35, 10], crs=ccrs.PlateCarree())
        
        # Características geográficas
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.STATES, alpha=0.3)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Campo de pressão
        contours = ax.contour(
            lon_mesh, lat_mesh, pressure_field,
            levels=np.arange(980, 1040, 4),
            colors='black',
            linewidths=0.8,
            alpha=0.7,
            transform=ccrs.PlateCarree()
        )
        ax.clabel(contours, inline=True, fontsize=8)
        
        # Plotar sistemas e trajetórias
        colors = {'HIGH': 'red', 'LOW': 'blue'}
        markers = {'HIGH': '^', 'LOW': 'v'}
        
        for system in systems:
            color = colors[system['type']]
            marker = markers[system['type']]
            
            # Sistema atual
            ax.plot(system['lon'], system['lat'], 
                   marker=marker, color=color, markersize=12,
                   transform=ccrs.PlateCarree())
            
            # ID do sistema
            ax.text(system['lon'] + 0.5, system['lat'] + 0.5, 
                   system['id'], fontsize=8, color=color,
                   transform=ccrs.PlateCarree())
            
            # Trajetória
            if 'track' in system and len(system['track']) > 1:
                track = np.array(system['track'])
                ax.plot(track[:, 1], track[:, 0], 
                       color=color, linewidth=2, alpha=0.7,
                       transform=ccrs.PlateCarree())
                
                # Pontos da trajetória
                ax.scatter(track[:-1, 1], track[:-1, 0], 
                          color=color, s=20, alpha=0.5,
                          transform=ccrs.PlateCarree())
            
            # Previsão de trajetória
            future_track = self.predict_trajectory(system, hours_ahead=12)
            if future_track:
                future_array = np.array(future_track)
                ax.plot(future_array[:, 1], future_array[:, 0], 
                       color=color, linewidth=2, linestyle='--', alpha=0.5,
                       transform=ccrs.PlateCarree())
        
        ax.gridlines(draw_labels=True, alpha=0.5)
        plt.title(f'Rastreamento de Sistemas Meteorológicos - {datetime.now().strftime("%d/%m/%Y %H:%M")}', 
                 fontsize=14, fontweight='bold')
        
        # Legenda
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                      markersize=10, label='Alta Pressão'),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='blue', 
                      markersize=10, label='Baixa Pressão'),
            plt.Line2D([0], [0], color='gray', linewidth=2, label='Trajetória'),
            plt.Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='Previsão')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Salvar na subpasta resultados_2/mapas
        if save_path:
            save_dir = os.path.join("resultados_systemTracker", "mapas")
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.basename(save_path)
            full_save_path = os.path.join(save_dir, filename)
            plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_tracking_report(self, systems, alerts):
        """
        Gera relatório de rastreamento
        """
        report = []
        report.append(f"RELATÓRIO DE RASTREAMENTO - {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        report.append("=" * 70)
        
        # Resumo de sistemas
        high_systems = [s for s in systems if s['type'] == 'HIGH']
        low_systems = [s for s in systems if s['type'] == 'LOW']
        
        report.append(f"\nSISTEMAS DETECTADOS:")
        report.append(f"Sistemas de Alta Pressão: {len(high_systems)}")
        report.append(f"Sistemas de Baixa Pressão: {len(low_systems)}")
        
        # Detalhes dos sistemas
        for system in systems:
            report.append(f"\n{system['id']} - {system['type']}")
            report.append(f"  Posição: {system['lat']:.2f}°S, {abs(system['lon']):.2f}°W")
            report.append(f"  Pressão: {system['pressure']:.1f} hPa")
            report.append(f"  Intensidade: {system['intensity']:.1f} hPa")
            
            if 'speed' in system:
                report.append(f"  Velocidade: {system['speed']:.1f} km/h")
            
            if 'track' in system:
                report.append(f"  Pontos rastreados: {len(system['track'])}")
        
        # Alertas
        if alerts:
            report.append(f"\nALERTAS METEOROLÓGICOS ({len(alerts)}):")
            for alert in alerts:
                report.append(f"- {alert['level']}: {alert['type']} em {alert['location']}")
                report.append(f"  Intensidade: {alert['intensity']:.1f} hPa")
        else:
            report.append(f"\nNenhum alerta ativo no momento.")
        
        return "\n".join(report)

# Simulação de rastreamento em tempo real
def run_tracking_simulation(tracker, time_steps=10, save_plots=True):
    """
    Executa simulação de rastreamento ao longo do tempo
    """
    print("Iniciando simulação de rastreamento...")
    
    all_reports = []
    
    for step in range(time_steps):
        print(f"\nProcessando time step {step + 1}/{time_steps}")
        
        # Gerar dados sintéticos
        lon_mesh, lat_mesh, pressure_field = tracker.generate_synthetic_pressure_field(step)
        temp_lon, temp_lat, temp_field = tracker.generate_synthetic_temperature_field(step)
        
        # Detectar sistemas
        systems = tracker.detect_pressure_systems(lon_mesh, lat_mesh, pressure_field)
        fronts = tracker.detect_fronts(temp_lon, temp_lat, temp_field)
        
        # Rastrear sistemas
        tracked_systems = tracker.track_systems(systems, step)
        
        # Gerar alertas
        alerts = tracker.generate_alerts(tracked_systems)
        
        # Visualizar (apenas alguns time steps para economizar recursos)
        if save_plots and step % 3 == 0:
            fig = tracker.visualize_tracking(
                lon_mesh, lat_mesh, pressure_field, tracked_systems,
                save_path=f"tracking_step_{step:02d}.png"
            )
            plt.close(fig)
        
        # Gerar relatório
        report = tracker.generate_tracking_report(tracked_systems, alerts)
        all_reports.append(report)
        
        if alerts:
            print(f"⚠️  {len(alerts)} alertas gerados!")
        
        print(f"Detectados {len(tracked_systems)} sistemas meteorológicos")
    
    return all_reports

def analyze_system_statistics(tracker):
    """
    Analisa estatísticas dos sistemas rastreados
    """
    if not tracker.systems_history:
        return "Nenhum sistema foi rastreado ainda."
    
    df = pd.DataFrame(tracker.systems_history)
    
    stats = []
    stats.append("ESTATÍSTICAS DE SISTEMAS METEOROLÓGICOS")
    stats.append("=" * 50)
    
    # Estatísticas por tipo
    for system_type in df['type'].unique():
        type_data = df[df['type'] == system_type]
        stats.append(f"\n{system_type} PRESSURE SYSTEMS:")
        stats.append(f"Total detectado: {len(type_data)}")
        stats.append(f"Intensidade média: {type_data['intensity'].mean():.2f} hPa")
        stats.append(f"Intensidade máxima: {type_data['intensity'].max():.2f} hPa")
        
        # Sistemas com trajetórias
        tracked = type_data[type_data['track'].apply(lambda x: len(x) > 1 if isinstance(x, list) else False)]
        if len(tracked) > 0:
            stats.append(f"Sistemas rastreados: {len(tracked)}")
            avg_speed = tracked['speed'].mean() if 'speed' in tracked.columns else 0
            stats.append(f"Velocidade média: {avg_speed:.1f} km/time_step")
    
    return "\n".join(stats)

# Exemplo de uso
if __name__ == "__main__":
    # Inicializar tracker
    tracker = WeatherSystemTracker()
    
    # Executar simulação
    reports = run_tracking_simulation(tracker, time_steps=8, save_plots=True)
    
    # Salvar relatórios na subpasta resultados_2/relatorios
    save_dir = os.path.join("resultados_systemTracker", "relatorios")
    os.makedirs(save_dir, exist_ok=True)
    relatorio_path = os.path.join(save_dir, "relatorios_rastreamento.txt")
    with open(relatorio_path, "w", encoding='utf-8') as f:
        for i, report in enumerate(reports):
            f.write(f"\n{'='*50}\n")
            f.write(f"TIME STEP {i+1}\n")
            f.write(f"{'='*50}\n")
            f.write(report)
            f.write("\n\n")
    
    # Análise estatística final
    final_stats = analyze_system_statistics(tracker)
    print("\n" + final_stats)
    
    stats_save_dir = os.path.join("resultados_systemTracker", "relatorios")
    os.makedirs(stats_save_dir, exist_ok=True)
    stats_path = os.path.join(stats_save_dir, "estatisticas_sistemas.txt")
    with open(stats_path, "w", encoding='utf-8') as f:
        f.write(final_stats)
    
    print(f"\nSimulação concluída! Gerados {len(reports)} relatórios.")
    print("Arquivos salvos:")
    print("- relatorios_rastreamento.txt")
    print("- estatisticas_sistemas.txt")
    print("- tracking_step_XX.png (gráficos)")