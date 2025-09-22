#!/usr/bin/env python3
"""
Script para analizar y visualizar los resultados de la comparación 
de diferentes configuraciones Epsilon-Greedy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import seaborn as sns
from datetime import datetime

# Configurar matplotlib para mejor apariencia
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_epsilon_comparison_data():
    """Cargar todos los datos de la comparación epsilon-greedy"""
    base_dir = Path(__file__).parent / "RESULTADOS" / "epsilon_greedy_comparison"
    
    if not base_dir.exists():
        print(f"❌ No se encontró el directorio: {base_dir}")
        return None
    
    all_data = []
    config_mapping = {
        'alta_exploracion': {'epsilon_start': 0.9, 'epsilon_min': 0.1, 'epsilon_decay': 0.999, 'label': 'Alta Exploración (ε=0.9)'},
        'moderada_alta': {'epsilon_start': 0.7, 'epsilon_min': 0.05, 'epsilon_decay': 0.9995, 'label': 'Moderada-Alta (ε=0.7)'},
        'balanceada': {'epsilon_start': 0.5, 'epsilon_min': 0.05, 'epsilon_decay': 0.9995, 'label': 'Balanceada (ε=0.5)'},
        'conservadora': {'epsilon_start': 0.3, 'epsilon_min': 0.01, 'epsilon_decay': 0.9998, 'label': 'Conservadora (ε=0.3)'},
        'muy_greedy': {'epsilon_start': 0.1, 'epsilon_min': 0.01, 'epsilon_decay': 0.9999, 'label': 'Muy Greedy (ε=0.1)'}
    }
    
    print("📊 Cargando datos de comparación epsilon-greedy...")
    
    for config_name, config_info in config_mapping.items():
        config_dir = base_dir / config_name
        if not config_dir.exists():
            print(f"⚠️ No se encontró configuración: {config_name}")
            continue
        
        config_data = []
        for run_dir in sorted(config_dir.iterdir(), key=lambda x: int(x.name)):
            if run_dir.is_dir():
                # Buscar archivos CSV
                csv_files = list(run_dir.glob("epsilon_greedy_summary_*.csv"))
                json_files = list(run_dir.glob("epsilon_greedy_raw_data_*.json"))
                
                if csv_files and json_files:
                    # Cargar CSV
                    csv_data = pd.read_csv(csv_files[0])
                    metrics_dict = dict(zip(csv_data['Métrica'], csv_data['Valor']))
                    
                    # Cargar JSON para detalles adicionales
                    with open(json_files[0], 'r') as f:
                        json_data = json.load(f)
                    
                    # Combinar datos
                    run_data = {
                        'configuracion': config_name,
                        'run': int(run_dir.name),
                        'pasos_totales': metrics_dict.get('Pasos Totales', 0),
                        'tiempo_segundos': metrics_dict.get('Tiempo (s)', 0),
                        'pasos_por_segundo': metrics_dict.get('Pasos/Segundo', 0),
                        'recompensa_total': metrics_dict.get('Recompensa Total', 0),
                        'recompensa_promedio': metrics_dict.get('Recompensa Promedio', 0),
                        'pokemon_obtenidos': metrics_dict.get('Pokemon Obtenidos', 0),
                        'suma_niveles': metrics_dict.get('Suma de Niveles', 0),
                        'posiciones_exploradas': metrics_dict.get('Posiciones Exploradas', 0),
                        'eficiencia_exploracion': metrics_dict.get('Eficiencia Exploración', 0),
                        'epsilon_start': config_info['epsilon_start'],
                        'epsilon_min': config_info['epsilon_min'],
                        'epsilon_decay': config_info['epsilon_decay'],
                        'label': config_info['label'],
                        'exito': metrics_dict.get('Pokemon Obtenidos', 0) >= 1,
                        'epsilon_final': json_data.get('epsilon_final', config_info['epsilon_min'])
                    }
                    
                    config_data.append(run_data)
        
        all_data.extend(config_data)
        print(f"  ✅ {config_name}: {len(config_data)} ejecuciones cargadas")
    
    if not all_data:
        print("❌ No se encontraron datos válidos")
        return None
    
    df = pd.DataFrame(all_data)
    print(f"📈 Total de datos cargados: {len(df)} ejecuciones")
    return df

def generate_epsilon_comparison_visualizations(df):
    """Generar visualizaciones comparativas de epsilon-greedy"""
    
    # Crear directorio de salida
    output_dir = Path(__file__).parent / "informe_visuals"
    output_dir.mkdir(exist_ok=True)
    
    # Configurar estilo
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # 1. COMPARACIÓN DE PASOS TOTALES
    plt.figure(figsize=(14, 8))
    df_grouped = df.groupby('label')['pasos_totales'].agg(['mean', 'std']).reset_index()
    
    bars = plt.bar(df_grouped['label'], df_grouped['mean'], 
                   yerr=df_grouped['std'], capsize=5, alpha=0.8,
                   color=['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728', '#9467bd'])
    
    plt.title('Comparación de Pasos Totales por Configuración Epsilon-Greedy', fontsize=16, fontweight='bold')
    plt.xlabel('Configuración de Epsilon', fontsize=12)
    plt.ylabel('Pasos Totales (Promedio ± DE)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + df_grouped['std'].iloc[i],
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'epsilon_pasos_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. COMPARACIÓN DE TIEMPO
    plt.figure(figsize=(14, 8))
    df_grouped_tiempo = df.groupby('label')['tiempo_segundos'].agg(['mean', 'std']).reset_index()
    
    bars = plt.bar(df_grouped_tiempo['label'], df_grouped_tiempo['mean'], 
                   yerr=df_grouped_tiempo['std'], capsize=5, alpha=0.8,
                   color=['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728', '#9467bd'])
    
    plt.title('Comparación de Tiempo de Ejecución por Configuración Epsilon-Greedy', fontsize=16, fontweight='bold')
    plt.xlabel('Configuración de Epsilon', fontsize=12)
    plt.ylabel('Tiempo (segundos, Promedio ± DE)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + df_grouped_tiempo['std'].iloc[i],
                f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'epsilon_tiempo_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. COMPARACIÓN DE RECOMPENSA TOTAL
    plt.figure(figsize=(14, 8))
    df_grouped_reward = df.groupby('label')['recompensa_total'].agg(['mean', 'std']).reset_index()
    
    bars = plt.bar(df_grouped_reward['label'], df_grouped_reward['mean'], 
                   yerr=df_grouped_reward['std'], capsize=5, alpha=0.8,
                   color=['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728', '#9467bd'])
    
    plt.title('Comparación de Recompensa Total por Configuración Epsilon-Greedy', fontsize=16, fontweight='bold')
    plt.xlabel('Configuración de Epsilon', fontsize=12)
    plt.ylabel('Recompensa Total (Promedio ± DE)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + df_grouped_reward['std'].iloc[i],
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'epsilon_recompensa_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. TASA DE ÉXITO
    plt.figure(figsize=(14, 8))
    success_rate = df.groupby('label')['exito'].mean() * 100
    
    bars = plt.bar(success_rate.index, success_rate.values, alpha=0.8,
                   color=['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728', '#9467bd'])
    
    plt.title('Tasa de Éxito por Configuración Epsilon-Greedy', fontsize=16, fontweight='bold')
    plt.xlabel('Configuración de Epsilon', fontsize=12)
    plt.ylabel('Tasa de Éxito (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    
    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'epsilon_exito_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. EFICIENCIA (Pasos por segundo)
    plt.figure(figsize=(14, 8))
    df_grouped_eff = df.groupby('label')['pasos_por_segundo'].agg(['mean', 'std']).reset_index()
    
    bars = plt.bar(df_grouped_eff['label'], df_grouped_eff['mean'], 
                   yerr=df_grouped_eff['std'], capsize=5, alpha=0.8,
                   color=['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728', '#9467bd'])
    
    plt.title('Eficiencia Computacional por Configuración Epsilon-Greedy', fontsize=16, fontweight='bold')
    plt.xlabel('Configuración de Epsilon', fontsize=12)
    plt.ylabel('Pasos por Segundo (Promedio ± DE)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + df_grouped_eff['std'].iloc[i],
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'epsilon_eficiencia_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Gráficos guardados en: {output_dir}")
    return output_dir

def generate_epsilon_statistics_table(df):
    """Generar tabla de estadísticas para LaTeX"""
    
    # Agrupar por configuración y calcular estadísticas
    stats = df.groupby('label').agg({
        'pasos_totales': ['mean', 'std'],
        'tiempo_segundos': ['mean', 'std'],
        'recompensa_total': ['mean', 'std'],
        'exito': 'mean',
        'pasos_por_segundo': ['mean', 'std']
    }).round(2)
    
    # Simplificar nombres de columnas
    stats.columns = ['Pasos_Mean', 'Pasos_Std', 'Tiempo_Mean', 'Tiempo_Std', 
                     'Reward_Mean', 'Reward_Std', 'Exito_Rate', 'PPS_Mean', 'PPS_Std']
    
    # Generar LaTeX
    latex_table = "\\begin{table}[H]\n\\centering\n\\caption{Comparación de Configuraciones Epsilon-Greedy}\n"
    latex_table += "\\label{tab:epsilon_comparison}\n\\begin{tabular}{|l|c|c|c|c|c|}\n\\hline\n"
    latex_table += "\\textbf{Configuración} & \\textbf{Pasos} & \\textbf{Tiempo (s)} & \\textbf{Recompensa} & \\textbf{Éxito (\\%)} & \\textbf{Pasos/s} \\\\\n\\hline\n"
    
    for idx, row in stats.iterrows():
        config_name = idx.replace(' (ε=', ' ($\\epsilon$=').replace(')', ')')
        latex_table += f"{config_name} & "
        latex_table += f"{row['Pasos_Mean']:.0f} ± {row['Pasos_Std']:.0f} & "
        latex_table += f"{row['Tiempo_Mean']:.2f} ± {row['Tiempo_Std']:.2f} & "
        latex_table += f"{row['Reward_Mean']:.1f} ± {row['Reward_Std']:.1f} & "
        latex_table += f"{row['Exito_Rate']*100:.1f} & "
        latex_table += f"{row['PPS_Mean']:.0f} ± {row['PPS_Std']:.0f} \\\\\n\\hline\n"
    
    latex_table += "\\end{tabular}\n\\end{table}\n"
    
    # Guardar tabla
    table_path = Path(__file__).parent / "informe_visuals" / "epsilon_comparison_table.tex"
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print(f"📋 Tabla LaTeX guardada en: {table_path}")
    return latex_table

def main():
    """Función principal"""
    print("🔬 Analizando resultados de comparación Epsilon-Greedy...")
    
    # Cargar datos
    df = load_epsilon_comparison_data()
    if df is None:
        return 1
    
    # Mostrar resumen
    print("\n📊 RESUMEN DE RESULTADOS:")
    print("="*50)
    summary = df.groupby('label').agg({
        'pasos_totales': 'mean',
        'tiempo_segundos': 'mean', 
        'recompensa_total': 'mean',
        'exito': 'mean'
    }).round(2)
    
    for idx, row in summary.iterrows():
        print(f"{idx}:")
        print(f"  Pasos promedio: {row['pasos_totales']:.0f}")
        print(f"  Tiempo promedio: {row['tiempo_segundos']:.2f}s")
        print(f"  Recompensa promedio: {row['recompensa_total']:.1f}")
        print(f"  Tasa de éxito: {row['exito']*100:.1f}%")
        print()
    
    # Generar visualizaciones
    output_dir = generate_epsilon_comparison_visualizations(df)
    
    # Generar tabla para LaTeX
    latex_table = generate_epsilon_statistics_table(df)
    
    print("✅ Análisis completado exitosamente!")
    print(f"📁 Archivos generados en: {output_dir}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())