import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# --- Configuración de Estilo Vanguardista ---
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# --- Rutas ---
RESULTS_DIR = Path('RESULTADOS')
VISUALS_DIR = Path('informe_visuals')
VISUALS_DIR.mkdir(exist_ok=True)

def get_algorithm_name_from_path(path):
    """Extrae un nombre de algoritmo limpio y estandarizado de la ruta del archivo."""
    parts = path.parts
    # Ejemplo: ('RESULTADOS', 'epsilon_greedy_comparison', 'alta_exploracion', '1', 'epsilon_greedy_summary_1758505057.csv')
    
    if 'ppo' in parts:
        return 'PPO'
    if 'epsilon_greedy_comparison' in parts:
        config_name = parts[parts.index('epsilon_greedy_comparison') + 1]
        epsilon_map = {
            'alta_exploracion': '0.9',
            'balanceada': '0.5',
            'conservadora': '0.3',
            'moderada_alta': '0.7',
            'muy_greedy': '0.1'
        }
        epsilon_val = epsilon_map.get(config_name, 'N/A')
        return f"Epsilon Greedy (e={epsilon_val})"
    if 'search_algorithms_comparison' in parts:
        algo_name = parts[parts.index('search_algorithms_comparison') + 1].replace('_', ' ').title()
        # Renombres para mayor claridad
        algo_name = algo_name.replace('Hill Climbing First Improvement', 'Hill Climbing (First Imp.)')
        algo_name = algo_name.replace('Hill Climbing Random Restart', 'Hill Climbing (Random Restart)')
        algo_name = algo_name.replace('Hill Climbing Steepest', 'Hill Climbing (Steepest)')
        algo_name = algo_name.replace('Breadth First Search', 'BFS')
        return algo_name
    
    # Fallback por si hay otras estructuras
    return parts[-3].replace('_', ' ').title()

def load_and_process_data(results_dir):
    """Carga, pivota y procesa todos los datos de resultados recursivamente, manejando nombres de columna correctos."""
    all_data = []
    summary_files = list(results_dir.rglob('*summary*.csv'))
    
    print(f"Encontrados {len(summary_files)} archivos de resumen para procesar.")

    # Columnas esenciales que esperamos después de pivotar, con los nombres correctos
    expected_cols = ['Pasos Totales', 'Tiempo (s)', 'Recompensa Total', 'Posiciones Exploradas']

    for file_path in summary_files:
        try:
            df_long = pd.read_csv(file_path)
            df_wide = df_long.set_index('Métrica').T
            df_wide.reset_index(drop=True, inplace=True)

            # Verificar y añadir columnas faltantes con NaN
            for col in expected_cols:
                if col not in df_wide.columns:
                    df_wide[col] = np.nan

            algo_name = get_algorithm_name_from_path(file_path)
            df_wide['Algorithm'] = algo_name
            
            all_data.append(df_wide)
        except Exception as e:
            print(f"Error al procesar o pivotar el archivo {file_path}: {e}")
    
    if not all_data:
        print("No se pudieron cargar y procesar datos de resumen. Saliendo.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Limpieza de datos: asegurar que las columnas numéricas lo sean
    for col in expected_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    # Eliminar filas donde las métricas CLAVE son nulas.
    combined_df.dropna(subset=['Pasos Totales', 'Tiempo (s)'], inplace=True)

    # Rellenar Recompensa Total con 0 si es NaN para el cálculo de eficiencia
    if 'Recompensa Total' in combined_df.columns:
        combined_df['Recompensa Total'].fillna(0, inplace=True)

    # Calcular métricas adicionales de eficiencia
    if 'Recompensa Total' in combined_df.columns and 'Pasos Totales' in combined_df.columns:
        combined_df['recompensa_por_paso'] = combined_df.apply(
            lambda row: row['Recompensa Total'] / row['Pasos Totales'] if row['Pasos Totales'] > 0 else 0,
            axis=1
        )
    else:
        print("Advertencia: No se pueden calcular métricas de eficiencia por falta de columnas.")
        
    return combined_df

def generate_master_comparison_plots(df):
    """Genera y guarda los gráficos comparativos maestros con un estilo mejorado."""
    if df.empty:
        print("El DataFrame está vacío. No se pueden generar gráficos.")
        return

    # Definir un orden categórico para los algoritmos basado en el tiempo de ejecución
    algo_order_time = df.groupby('Algorithm')['Tiempo (s)'].mean().sort_values().index
    
    # Paleta de colores
    palette = sns.color_palette("plasma", n_colors=len(algo_order_time))

    # 1. Gráfico de Tiempo de Ejecución (Logarítmico)
    plt.figure(figsize=(14, 10))
    sns.barplot(data=df, x='Tiempo (s)', y='Algorithm', order=algo_order_time, palette=palette, orient='h', ci=None)
    plt.xscale('log')
    plt.title('Comparación Global: Tiempo de Ejecución Promedio (Escala Logarítmica)')
    plt.xlabel('Tiempo Promedio de Ejecución (segundos) - Escala Log')
    plt.ylabel('Algoritmo')
    
    # Añadir etiquetas de valor
    for i, algorithm in enumerate(algo_order_time):
        mean_val = df[df['Algorithm'] == algorithm]['Tiempo (s)'].mean()
        plt.text(mean_val * 1.1, i, f'{mean_val:.3f} s', color='black', ha="left", va='center', fontsize=9)
        
    plt.tight_layout(pad=2)
    plt.savefig(VISUALS_DIR / 'master_tiempo_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico 'master_tiempo_comparacion.png' guardado.")

    # 2. Gráfico de Pasos Totales
    algo_order_steps = df.groupby('Algorithm')['Pasos Totales'].mean().sort_values().index
    plt.figure(figsize=(14, 10))
    sns.barplot(data=df, x='Pasos Totales', y='Algorithm', order=algo_order_steps, palette=palette, orient='h', ci=None)
    plt.title('Comparación Global: Pasos Totales Promedio')
    plt.xlabel('Pasos Totales Promedio')
    plt.ylabel('Algoritmo')

    for i, algorithm in enumerate(algo_order_steps):
        mean_val = df[df['Algorithm'] == algorithm]['Pasos Totales'].mean()
        plt.text(mean_val * 1.02, i, f'{mean_val:.1f}', color='black', ha="left", va='center', fontsize=9)

    plt.tight_layout(pad=2)
    plt.savefig(VISUALS_DIR / 'master_pasos_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico 'master_pasos_comparacion.png' guardado.")

    # 3. Gráfico de Eficiencia (Recompensa por Paso)
    algo_order_eff = df.groupby('Algorithm')['recompensa_por_paso'].mean().sort_values(ascending=False).index
    plt.figure(figsize=(14, 10))
    sns.barplot(data=df, x='recompensa_por_paso', y='Algorithm', order=algo_order_eff, palette=palette, orient='h', ci=None)
    plt.title('Comparación Global: Eficiencia Promedio (Recompensa por Paso)')
    plt.xlabel('Recompensa Promedio por Paso')
    plt.ylabel('Algoritmo')

    for i, algorithm in enumerate(algo_order_eff):
        mean_val = df[df['Algorithm'] == algorithm]['recompensa_por_paso'].mean()
        plt.text(mean_val * 1.02, i, f'{mean_val:.2f}', color='black', ha="left", va='center', fontsize=9)

    plt.tight_layout(pad=2)
    plt.savefig(VISUALS_DIR / 'master_eficiencia_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico 'master_eficiencia_comparacion.png' guardado.")

def generate_master_latex_table(df):
    """Genera una tabla LaTeX con las métricas clave y la guarda en un archivo .tex."""
    if df.empty:
        print("El DataFrame está vacío. No se puede generar la tabla LaTeX.")
        return

    # Agrupar y calcular métricas promedio y de desviación estándar
    summary_df = df.groupby('Algorithm').agg(
        avg_time=('Tiempo (s)', 'mean'),
        std_time=('Tiempo (s)', 'std'),
        avg_steps=('Pasos Totales', 'mean'),
        std_steps=('Pasos Totales', 'std'),
        avg_reward_per_step=('recompensa_por_paso', 'mean'),
        std_reward_per_step=('recompensa_por_paso', 'std'),
        runs=('Algorithm', 'size') # Contar el número de ejecuciones
    ).reset_index()

    # Ordenar por la métrica principal de interés, por ejemplo, tiempo promedio
    summary_df = summary_df.sort_values('avg_time')

    # Rellenar NaNs en las desviaciones estándar (para grupos con una sola muestra)
    summary_df.fillna(0, inplace=True)

    # Formatear los números para la tabla
    summary_df['avg_time_str'] = summary_df['avg_time'].map('{:.2f}'.format)
    summary_df['std_time_str'] = summary_df['std_time'].map('{:.2f}'.format)
    summary_df['avg_steps_str'] = summary_df['avg_steps'].map('{:.1f}'.format)
    summary_df['std_steps_str'] = summary_df['std_steps'].map('{:.1f}'.format)
    summary_df['avg_reward_per_step_str'] = summary_df['avg_reward_per_step'].map('{:.3f}'.format)
    summary_df['std_reward_per_step_str'] = summary_df['std_reward_per_step'].map('{:.3f}'.format)

    # Combinar promedio y desviación estándar en una sola columna
    summary_df['Tiempo (s)'] = summary_df['avg_time_str'] + ' ± ' + summary_df['std_time_str']
    summary_df['Pasos Totales'] = summary_df['avg_steps_str'] + ' ± ' + summary_df['std_steps_str']
    summary_df['Eficiencia (Rew/Paso)'] = summary_df['avg_reward_per_step_str'] + ' ± ' + summary_df['std_reward_per_step_str']
    
    # Seleccionar y renombrar columnas para la salida final
    final_table = summary_df[['Algorithm', 'Tiempo (s)', 'Pasos Totales', 'Eficiencia (Rew/Paso)', 'runs']]
    final_table = final_table.rename(columns={'Algorithm': 'Algoritmo', 'runs': 'N° Ejs.'})

    # Generar el código LaTeX
    latex_code = final_table.to_latex(index=False,
                                      header=True,
                                      column_format='l|r|r|r|c',
                                      longtable=False,
                                      escape=True)
    
    # Envolver en un entorno table con caption, label y resizebox
    final_latex_code = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Comparación Agregada de Métricas de Algoritmos.}\n"
        "\\label{tab:master_comparison}\n"
        "\\resizebox{\\textwidth}{!}{\n"
        f"{latex_code}"
        "}\n"
        "\\end{table}\n"
    )
    
    # Reemplazar los separadores de línea de booktabs por líneas verticales completas
    final_latex_code = final_latex_code.replace('\\toprule', '\\hline')
    final_latex_code = final_latex_code.replace('\\midrule', '\\hline')
    final_latex_code = final_latex_code.replace('\\bottomrule', '\\hline')

    # Guardar en archivo
    table_path = VISUALS_DIR / 'master_comparison_table.tex'
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(final_latex_code)
    
    print(f"Tabla LaTeX 'master_comparison_table.tex' guardada.")


if __name__ == '__main__':
    master_df = load_and_process_data(RESULTS_DIR)
    
    if not master_df.empty:
        generate_master_comparison_plots(master_df)
        generate_master_latex_table(master_df)
        print("\nProceso completado. Se han generado los nuevos gráficos y la tabla LaTeX.")
    else:
        print("\nEl proceso no pudo completarse debido a la falta de datos.")

