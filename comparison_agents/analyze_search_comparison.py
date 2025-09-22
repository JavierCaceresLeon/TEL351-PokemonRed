#!/usr/bin/env python3
"""
Análisis y visualización de resultados de comparación de algoritmos de búsqueda
en Pokemon Red.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para mejor visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_search_comparison_data():
    """Cargar datos de la comparación de algoritmos de búsqueda"""
    
    base_dir = Path("RESULTADOS/search_algorithms_comparison")
    if not base_dir.exists():
        print(f"❌ No se encontró el directorio de resultados: {base_dir}")
        return None
    
    all_data = []
    algorithm_configs = {}
    
    # Mapeo de nombres de algoritmos para visualización
    algorithm_labels = {
        'astar_default': 'A* Search',
        'bfs_default': 'Breadth-First Search',
        'simulated_annealing_default': 'Simulated Annealing',
        'hill_climbing_steepest_ascent': 'Hill Climbing (Steepest)',
        'hill_climbing_first_improvement': 'Hill Climbing (First Imp.)',
        'hill_climbing_random_restart': 'Hill Climbing (Random Restart)',
        'tabu_search_default': 'Tabu Search'
    }
    
    print("📊 Cargando datos de algoritmos de búsqueda...")
    
    for algorithm_dir in base_dir.iterdir():
        if not algorithm_dir.is_dir():
            continue
        
        algorithm_name = algorithm_dir.name
        algorithm_label = algorithm_labels.get(algorithm_name, algorithm_name.replace('_', ' ').title())
        
        print(f"  📁 Procesando: {algorithm_label}")
        
        # Cargar configuración del algoritmo
        config_file = algorithm_dir / 'config_summary.json'
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                algorithm_configs[algorithm_name] = json.load(f)
        
        # Cargar datos de cada ejecución
        algorithm_data = []
        for run_dir in algorithm_dir.iterdir():
            if not run_dir.is_dir() or not run_dir.name.isdigit():
                continue
            
            # Buscar archivos CSV de resumen
            csv_files = list(run_dir.glob("*_summary_*.csv"))
            if csv_files:
                try:
                    csv_data = pd.read_csv(csv_files[0])
                    metrics_dict = dict(zip(csv_data['Métrica'], csv_data['Valor']))
                    
                    run_data = {
                        'algorithm': algorithm_name,
                        'algorithm_label': algorithm_label,
                        'run_number': int(run_dir.name),
                        'pasos_totales': float(metrics_dict.get('Pasos Totales', 0)),
                        'tiempo_segundos': float(metrics_dict.get('Tiempo (s)', 0)),
                        'recompensa_total': float(metrics_dict.get('Recompensa Total', 0)),
                        'pokemon_obtenidos': int(metrics_dict.get('Pokemon Obtenidos', 0)),
                        'posiciones_visitadas': int(metrics_dict.get('Posiciones Visitadas', 0)),
                        'eficiencia': float(metrics_dict.get('Eficiencia (Recompensa/Paso)', 0)),
                        'velocidad': float(metrics_dict.get('Velocidad (Pasos/s)', 0)),
                        'razon_terminacion': str(metrics_dict.get('Razón de Terminación', 'Desconocida'))
                    }
                    algorithm_data.append(run_data)
                    
                except Exception as e:
                    print(f"    ⚠️  Error procesando {run_dir}: {e}")
        
        all_data.extend(algorithm_data)
        print(f"    ✅ Cargadas {len(algorithm_data)} ejecuciones")
    
    if not all_data:
        print("❌ No se encontraron datos válidos")
        return None
    
    df = pd.DataFrame(all_data)
    print(f"\n📈 Total de datos cargados: {len(df)} ejecuciones de {df['algorithm_label'].nunique()} algoritmos")
    
    return df

def generate_search_comparison_visualizations(df, output_dir):
    """Generar visualizaciones comparativas de algoritmos de búsqueda"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar estilo
    colors = plt.cm.Set3(np.linspace(0, 1, df['algorithm_label'].nunique()))
    
    print("🎨 Generando visualizaciones...")
    
    # 1. Comparación de Pasos Totales
    plt.figure(figsize=(14, 8))
    df_grouped = df.groupby('algorithm_label')['pasos_totales'].agg(['mean', 'std']).reset_index()
    
    bars = plt.bar(df_grouped['algorithm_label'], df_grouped['mean'], 
                   yerr=df_grouped['std'], capsize=5, alpha=0.8, color=colors)
    
    plt.title('Comparación de Pasos Totales por Algoritmo de Búsqueda', fontsize=16, fontweight='bold')
    plt.xlabel('Algoritmo de Búsqueda', fontsize=12)
    plt.ylabel('Pasos Totales (Promedio ± DE)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + df_grouped['std'].iloc[i],
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'search_pasos_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Comparación de Tiempo de Ejecución
    plt.figure(figsize=(14, 8))
    df_grouped_time = df.groupby('algorithm_label')['tiempo_segundos'].agg(['mean', 'std']).reset_index()
    
    bars = plt.bar(df_grouped_time['algorithm_label'], df_grouped_time['mean'], 
                   yerr=df_grouped_time['std'], capsize=5, alpha=0.8, color=colors)
    
    plt.title('Comparación de Tiempo de Ejecución por Algoritmo', fontsize=16, fontweight='bold')
    plt.xlabel('Algoritmo de Búsqueda', fontsize=12)
    plt.ylabel('Tiempo (segundos) (Promedio ± DE)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + df_grouped_time['std'].iloc[i],
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'search_tiempo_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Comparación de Recompensa Total
    plt.figure(figsize=(14, 8))
    df_grouped_reward = df.groupby('algorithm_label')['recompensa_total'].agg(['mean', 'std']).reset_index()
    
    bars = plt.bar(df_grouped_reward['algorithm_label'], df_grouped_reward['mean'], 
                   yerr=df_grouped_reward['std'], capsize=5, alpha=0.8, color=colors)
    
    plt.title('Comparación de Recompensa Total por Algoritmo', fontsize=16, fontweight='bold')
    plt.xlabel('Algoritmo de Búsqueda', fontsize=12)
    plt.ylabel('Recompensa Total (Promedio ± DE)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + df_grouped_reward['std'].iloc[i],
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'search_recompensa_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Tasa de Éxito
    plt.figure(figsize=(14, 8))
    success_rate = df.groupby('algorithm_label')['pokemon_obtenidos'].agg(['mean', 'count']).reset_index()
    success_rate['success_percentage'] = success_rate['mean'] * 100
    
    bars = plt.bar(success_rate['algorithm_label'], success_rate['success_percentage'], 
                   alpha=0.8, color=colors)
    
    plt.title('Tasa de Éxito por Algoritmo de Búsqueda', fontsize=16, fontweight='bold')
    plt.xlabel('Algoritmo de Búsqueda', fontsize=12)
    plt.ylabel('Tasa de Éxito (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'search_exito_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Eficiencia (Recompensa/Paso)
    plt.figure(figsize=(14, 8))
    df_grouped_eff = df.groupby('algorithm_label')['eficiencia'].agg(['mean', 'std']).reset_index()
    
    bars = plt.bar(df_grouped_eff['algorithm_label'], df_grouped_eff['mean'], 
                   yerr=df_grouped_eff['std'], capsize=5, alpha=0.8, color=colors)
    
    plt.title('Eficiencia por Algoritmo de Búsqueda', fontsize=16, fontweight='bold')
    plt.xlabel('Algoritmo de Búsqueda', fontsize=12)
    plt.ylabel('Eficiencia (Recompensa/Paso) (Promedio ± DE)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + df_grouped_eff['std'].iloc[i],
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'search_eficiencia_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Comparación múltiple (subplot)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Pasos
    df.boxplot(column='pasos_totales', by='algorithm_label', ax=ax1)
    ax1.set_title('Distribución de Pasos Totales')
    ax1.set_xlabel('Algoritmo')
    ax1.set_ylabel('Pasos')
    ax1.tick_params(axis='x', rotation=45)
    
    # Tiempo
    df.boxplot(column='tiempo_segundos', by='algorithm_label', ax=ax2)
    ax2.set_title('Distribución de Tiempo')
    ax2.set_xlabel('Algoritmo')
    ax2.set_ylabel('Segundos')
    ax2.tick_params(axis='x', rotation=45)
    
    # Recompensa
    df.boxplot(column='recompensa_total', by='algorithm_label', ax=ax3)
    ax3.set_title('Distribución de Recompensa')
    ax3.set_xlabel('Algoritmo')
    ax3.set_ylabel('Recompensa')
    ax3.tick_params(axis='x', rotation=45)
    
    # Eficiencia
    df.boxplot(column='eficiencia', by='algorithm_label', ax=ax4)
    ax4.set_title('Distribución de Eficiencia')
    ax4.set_xlabel('Algoritmo')
    ax4.set_ylabel('Eficiencia')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Distribuciones Comparativas de Algoritmos de Búsqueda', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'search_distribuciones_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizaciones guardadas en {output_dir}")

def generate_search_statistics_table(df, output_dir):
    """Generar tabla estadística de algoritmos de búsqueda"""
    
    output_dir = Path(output_dir)
    
    # Calcular estadísticas por algoritmo
    stats = df.groupby('algorithm_label').agg({
        'pasos_totales': ['mean', 'std', 'min', 'max'],
        'tiempo_segundos': ['mean', 'std'],
        'recompensa_total': ['mean', 'std'],
        'pokemon_obtenidos': ['mean', 'count'],
        'eficiencia': ['mean', 'std'],
        'velocidad': ['mean', 'std']
    }).round(2)
    
    # Crear tabla LaTeX
    latex_content = """\\begin{table}[H]
\\centering
\\caption{Comparación de Algoritmos de Búsqueda}
\\label{tab:search_comparison}
\\begin{tabular}{|l|c|c|c|c|c|}
\\hline
\\textbf{Algoritmo} & \\textbf{Pasos} & \\textbf{Tiempo (s)} & \\textbf{Recompensa} & \\textbf{Éxito (\\%)} & \\textbf{Eficiencia} \\\\
\\hline
"""
    
    for algorithm in stats.index:
        pasos_mean = stats.loc[algorithm, ('pasos_totales', 'mean')]
        pasos_std = stats.loc[algorithm, ('pasos_totales', 'std')]
        tiempo_mean = stats.loc[algorithm, ('tiempo_segundos', 'mean')]
        tiempo_std = stats.loc[algorithm, ('tiempo_segundos', 'std')]
        reward_mean = stats.loc[algorithm, ('recompensa_total', 'mean')]
        reward_std = stats.loc[algorithm, ('recompensa_total', 'std')]
        success_rate = stats.loc[algorithm, ('pokemon_obtenidos', 'mean')] * 100
        eff_mean = stats.loc[algorithm, ('eficiencia', 'mean')]
        eff_std = stats.loc[algorithm, ('eficiencia', 'std')]
        
        latex_content += f"{algorithm} & {pasos_mean:.0f} ± {pasos_std:.0f} & {tiempo_mean:.2f} ± {tiempo_std:.2f} & {reward_mean:.1f} ± {reward_std:.1f} & {success_rate:.1f} & {eff_mean:.4f} ± {eff_std:.4f} \\\\\n\\hline\n"
    
    latex_content += """\\end{tabular}
\\end{table}"""
    
    # Guardar tabla LaTeX
    with open(output_dir / 'search_comparison_table.tex', 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    # Crear resumen en CSV
    summary_stats = pd.DataFrame({
        'Algoritmo': stats.index,
        'Pasos_Promedio': stats[('pasos_totales', 'mean')].values,
        'Pasos_Desviacion': stats[('pasos_totales', 'std')].values,
        'Tiempo_Promedio': stats[('tiempo_segundos', 'mean')].values,
        'Tiempo_Desviacion': stats[('tiempo_segundos', 'std')].values,
        'Recompensa_Promedio': stats[('recompensa_total', 'mean')].values,
        'Recompensa_Desviacion': stats[('recompensa_total', 'std')].values,
        'Tasa_Exito': (stats[('pokemon_obtenidos', 'mean')] * 100).values,
        'Eficiencia_Promedio': stats[('eficiencia', 'mean')].values,
        'Eficiencia_Desviacion': stats[('eficiencia', 'std')].values
    })
    
    summary_stats.to_csv(output_dir / 'search_comparison_summary.csv', index=False)
    
    print("📊 Tabla estadística generada")
    print("\nRanking por eficiencia:")
    ranking = summary_stats.sort_values('Eficiencia_Promedio', ascending=False)
    for i, (_, row) in enumerate(ranking.iterrows(), 1):
        print(f"  {i}. {row['Algoritmo']}: {row['Eficiencia_Promedio']:.4f} eficiencia")

def main():
    """Función principal de análisis"""
    
    print("🔍 Análisis de Comparación de Algoritmos de Búsqueda")
    print("=" * 60)
    
    # Cargar datos
    df = load_search_comparison_data()
    if df is None:
        print("❌ No se pudieron cargar los datos. Ejecutar primero run_complete_search_comparison.py")
        return False
    
    # Directorio de salida
    output_dir = Path("informe_visuals")
    output_dir.mkdir(exist_ok=True)
    
    # Generar análisis
    print("\n📈 Generando estadísticas descriptivas...")
    print(df.groupby('algorithm_label')[['pasos_totales', 'tiempo_segundos', 'recompensa_total', 'eficiencia']].describe())
    
    # Generar visualizaciones
    print("\n🎨 Generando visualizaciones...")
    generate_search_comparison_visualizations(df, output_dir)
    
    # Generar tabla estadística
    print("\n📊 Generando tabla estadística...")
    generate_search_statistics_table(df, output_dir)
    
    print(f"\n✅ Análisis completado. Resultados en {output_dir}")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"💥 Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)