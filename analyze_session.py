#!/usr/bin/env python3
"""
Script para analizar sesiones de Pokémon Red RL
Uso: python analyze_session.py [directorio_de_sesión]
"""

import sys
import pandas as pd
import json
import gzip
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_session(session_dir):
    """Analiza una sesión de entrenamiento de Pokémon Red"""
    session_path = Path(session_dir)
    
    if not session_path.exists():
        print(f"❌ Directorio {session_dir} no encontrado")
        return
    
    print(f"🔍 Analizando sesión: {session_path.name}")
    print("=" * 50)
    
    # Buscar archivos de estadísticas
    stats_files = list(session_path.glob("agent_stats_*.csv.gz"))
    if not stats_files:
        print("❌ No se encontraron archivos de estadísticas (agent_stats_*.csv.gz)")
        return
    
    # Cargar estadísticas
    stats_file = stats_files[0]
    print(f"📊 Cargando: {stats_file.name}")
    
    try:
        df = pd.read_csv(stats_file, compression='gzip')
    except Exception as e:
        print(f"❌ Error cargando estadísticas: {e}")
        return
    
    # Análisis básico
    print(f"\n📈 Estadísticas Básicas:")
    print(f"  • Total de pasos: {df['step'].max():,}")
    print(f"  • Duración: ~{df['step'].max() / 60:.1f} minutos de juego")
    print(f"  • Ubicaciones únicas: {df[['x', 'y', 'map']].drop_duplicates().shape[0]:,}")
    
    if 'levels_sum' in df.columns:
        print(f"  • Nivel máximo alcanzado: {df['levels_sum'].max()}")
    
    if 'badge' in df.columns:
        print(f"  • Medallas obtenidas: {df['badge'].max()}")
    
    if 'deaths' in df.columns:
        print(f"  • Muertes: {df['deaths'].max()}")
    
    # Mapas visitados
    maps_visited = df['map'].unique()
    print(f"\n🗺️ Mapas visitados ({len(maps_visited)}):")
    map_counts = df['map'].value_counts().head(10)
    for map_id, count in map_counts.items():
        map_name = df[df['map'] == map_id]['map_location'].iloc[0] if 'map_location' in df.columns else f"Mapa {map_id}"
        print(f"  • {map_name}: {count:,} pasos")
    
    # Progreso de exploración
    if 'coord_count' in df.columns:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(df['step'], df['coord_count'])
        plt.title('Progreso de Exploración')
        plt.xlabel('Pasos')
        plt.ylabel('Coordenadas Únicas')
        
        if 'levels_sum' in df.columns:
            plt.subplot(1, 3, 2)
            plt.plot(df['step'], df['levels_sum'])
            plt.title('Progreso de Niveles')
            plt.xlabel('Pasos')
            plt.ylabel('Suma de Niveles')
        
        if 'hp' in df.columns:
            plt.subplot(1, 3, 3)
            plt.plot(df['step'], df['hp'])
            plt.title('Salud del Party')
            plt.xlabel('Pasos')
            plt.ylabel('HP Fracción')
        
        plt.tight_layout()
        output_path = session_path / 'analysis_plot.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Gráfico guardado en: {output_path}")
        plt.show()
    
    # Buscar archivo de resumen
    json_files = list(session_path.glob("all_runs_*.json"))
    if json_files:
        json_file = json_files[0]
        try:
            with open(json_file, 'r') as f:
                runs_data = json.load(f)
            print(f"\n🏆 Resumen de ejecuciones ({len(runs_data)} runs):")
            if runs_data:
                last_run = runs_data[-1]
                for key, value in last_run.items():
                    print(f"  • {key}: {value:.2f}")
        except Exception as e:
            print(f"❌ Error cargando resumen: {e}")
    
    # Contar screenshots
    screenshots = list(session_path.glob("curframe_*.jpeg"))
    print(f"\n📸 Screenshots disponibles: {len(screenshots)}")
    
    final_states_dir = session_path / 'final_states'
    if final_states_dir.exists():
        final_screenshots = list(final_states_dir.glob("*.jpeg"))
        print(f"🎯 Estados finales: {len(final_screenshots)}")

def main():
    if len(sys.argv) < 2:
        print("Uso: python analyze_session.py [directorio_de_sesión]")
        print("\nEjemplo:")
        print("  python analyze_session.py v2/session_752558fa")
        print("  python analyze_session.py session_752558fa  # Si ya estás en v2/")
        return
    
    session_dir = sys.argv[1]
    analyze_session(session_dir)

if __name__ == "__main__":
    main()
