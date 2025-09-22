
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import json

# --- Configuración de Rutas ---
base_path = Path('c:/Users/Cris/Documents/GitHub/TEL351-PokemonRed/RESULTADOS')
ppo_path = base_path / 'ppo' / 'ppo'
epsilon_path = base_path / 'epsilon_greedy' / 'epsilon_greedy'
epsilon_comparison_path = base_path / 'epsilon_greedy_comparison'
output_path = Path('c:/Users/Cris/Documents/GitHub/TEL351-PokemonRed/informe_visuals')

# --- Función para Cargar Datos ---
def load_metric_from_csv(folder_path, metric_name):
    """Carga una métrica específica de todos los archivos CSV de resumen en una carpeta."""
    values = []
    # Ordenar los archivos para una consistencia en la lectura
    files = sorted(folder_path.glob('**/*_summary_*.csv'), key=lambda p: int(p.parent.name))
    for file in files:
        try:
            df = pd.read_csv(file, index_col=0)
            # El valor está en la segunda columna (índice 1)
            value = df.loc[metric_name].iloc[0]
            values.append(float(value))
        except (FileNotFoundError, KeyError, IndexError) as e:
            print(f"Advertencia: No se pudo procesar {file}. Error: {e}")
    return values

def load_epsilon_comparison_data():
    """Cargar datos de la comparación de configuraciones epsilon"""
    if not epsilon_comparison_path.exists():
        print(f"❌ No se encontró el directorio de comparación epsilon: {epsilon_comparison_path}")
        return None
    
    all_data = []
    config_mapping = {
        'alta_exploracion': {'epsilon_start': 0.9, 'label': 'Alta Exploración (ε=0.9)'},
        'moderada_alta': {'epsilon_start': 0.7, 'label': 'Moderada-Alta (ε=0.7)'},
        'balanceada': {'epsilon_start': 0.5, 'label': 'Balanceada (ε=0.5)'},
        'conservadora': {'epsilon_start': 0.3, 'label': 'Conservadora (ε=0.3)'},
        'muy_greedy': {'epsilon_start': 0.1, 'label': 'Muy Greedy (ε=0.1)'}
    }
    
    for config_name, config_info in config_mapping.items():
        config_dir = epsilon_comparison_path / config_name
        if not config_dir.exists():
            continue
        
        config_data = []
        for run_dir in sorted(config_dir.iterdir(), key=lambda x: int(x.name)):
            if run_dir.is_dir():
                csv_files = list(run_dir.glob("epsilon_greedy_summary_*.csv"))
                if csv_files:
                    csv_data = pd.read_csv(csv_files[0])
                    metrics_dict = dict(zip(csv_data['Métrica'], csv_data['Valor']))
                    
                    run_data = {
                        'configuracion': config_name,
                        'label': config_info['label'],
                        'epsilon_start': config_info['epsilon_start'],
                        'pasos_totales': metrics_dict.get('Pasos Totales', 0),
                        'tiempo_segundos': metrics_dict.get('Tiempo (s)', 0),
                        'recompensa_total': metrics_dict.get('Recompensa Total', 0),
                        'pokemon_obtenidos': metrics_dict.get('Pokemon Obtenidos', 0)
                    }
                    config_data.append(run_data)
        
        all_data.extend(config_data)
    
    return pd.DataFrame(all_data) if all_data else None

# --- Carga de Datos ---
ppo_steps = load_metric_from_csv(ppo_path, 'Pasos Totales')
ppo_time = load_metric_from_csv(ppo_path, 'Tiempo (s)')

epsilon_steps = load_metric_from_csv(epsilon_path, 'Pasos Totales')
epsilon_time = load_metric_from_csv(epsilon_path, 'Tiempo (s)')

# Cargar datos de comparación epsilon
epsilon_comparison_df = load_epsilon_comparison_data()

# Asegurarse de que los datos tengan la misma longitud para la comparación
num_runs = min(len(ppo_steps), len(epsilon_steps))
run_labels = [f'Run {i+1}' for i in range(num_runs)]

ppo_steps = ppo_steps[:num_runs]
ppo_time = ppo_time[:num_runs]
epsilon_steps = epsilon_steps[:num_runs]
epsilon_time = epsilon_time[:num_runs]

# --- Cálculo de Estadísticas ---
stats = {
    "PPO": {
        "Pasos Promedio": np.mean(ppo_steps),
        "Desviación Estándar (Pasos)": np.std(ppo_steps),
        "Tiempo Promedio (s)": np.mean(ppo_time),
        "Desviación Estándar (Tiempo)": np.std(ppo_time),
    },
    "Epsilon-Greedy": {
        "Pasos Promedio": np.mean(epsilon_steps),
        "Desviación Estándar (Pasos)": np.std(epsilon_steps),
        "Tiempo Promedio (s)": np.mean(epsilon_time),
        "Desviación Estándar (Tiempo)": np.std(epsilon_time),
    }
}

print("Estadísticas Descriptivas:")
print(pd.DataFrame(stats).to_latex())

# Estadísticas de comparación epsilon
if epsilon_comparison_df is not None:
    print("\nEstadísticas de Configuraciones Epsilon-Greedy:")
    epsilon_stats = epsilon_comparison_df.groupby('label').agg({
        'pasos_totales': ['mean', 'std'],
        'tiempo_segundos': ['mean', 'std'],
        'recompensa_total': ['mean', 'std']
    }).round(2)
    print(epsilon_stats)

# --- Generación de Gráficos ---

# 1. Gráfico de Pasos Totales
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(num_runs)
width = 0.35

rects1 = ax.bar(x - width/2, ppo_steps, width, label='PPO', color='cornflowerblue')
rects2 = ax.bar(x + width/2, epsilon_steps, width, label='Epsilon-Greedy', color='salmon')

ax.set_ylabel('Pasos Totales')
ax.set_title('Comparación de Pasos Totales por Ejecución')
ax.set_xticks(x)
ax.set_xticklabels(run_labels, rotation=45, ha="right")
ax.legend()
ax.set_yscale('log') # Escala logarítmica por la gran diferencia
ax.bar_label(rects1, padding=3, fmt='%d')
ax.bar_label(rects2, padding=3, fmt='%d', rotation=90)

fig.tight_layout()
plt.savefig(output_path / 'pasos_comparacion.png', dpi=300)
print(f"Gráfico de pasos guardado en {output_path / 'pasos_comparacion.png'}")

# 2. Gráfico de Tiempo Total
fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - width/2, ppo_time, width, label='PPO', color='cornflowerblue')
rects2 = ax.bar(x + width/2, epsilon_time, width, label='Epsilon-Greedy', color='salmon')

ax.set_ylabel('Tiempo Total (s)')
ax.set_title('Comparación de Tiempo Total (s) por Ejecución')
ax.set_xticks(x)
ax.set_xticklabels(run_labels, rotation=45, ha="right")
ax.legend()
ax.set_yscale('log') # Escala logarítmica
ax.bar_label(rects1, padding=3, fmt='%.2f')
ax.bar_label(rects2, padding=3, fmt='%.2f', rotation=90)

fig.tight_layout()
plt.savefig(output_path / 'tiempo_comparacion.png', dpi=300)
print(f"Gráfico de tiempo guardado en {output_path / 'tiempo_comparacion.png'}")

# 3. Gráfico adicional: Comparación de configuraciones epsilon (si están disponibles)
if epsilon_comparison_df is not None:
    print("\n📊 Generando gráfico adicional de comparación epsilon...")
    
    # Gráfico de pasos por configuración epsilon
    fig, ax = plt.subplots(figsize=(14, 8))
    df_grouped = epsilon_comparison_df.groupby('label')['pasos_totales'].agg(['mean', 'std']).reset_index()
    
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728', '#9467bd']
    bars = ax.bar(df_grouped['label'], df_grouped['mean'], 
                  yerr=df_grouped['std'], capsize=5, alpha=0.8, color=colors)
    
    ax.set_title('Comparación de Pasos Totales por Configuración Epsilon-Greedy', fontsize=16, fontweight='bold')
    ax.set_xlabel('Configuración de Epsilon', fontsize=12)
    ax.set_ylabel('Pasos Totales (Promedio ± DE)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + df_grouped['std'].iloc[i],
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'epsilon_configs_pasos_comparacion.png', dpi=300, bbox_inches='tight')
    print(f"Gráfico de configuraciones epsilon guardado en {output_path / 'epsilon_configs_pasos_comparacion.png'}")

print(f"\n✅ Todos los gráficos generados en: {output_path}")
