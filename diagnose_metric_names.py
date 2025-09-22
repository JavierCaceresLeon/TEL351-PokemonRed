import pandas as pd
from pathlib import Path

RESULTS_DIR = Path('RESULTADOS')

def diagnose_metric_names(results_dir):
    """Carga todos los archivos de resumen e imprime todos los nombres de métricas únicos."""
    all_metrics = set()
    summary_files = list(results_dir.rglob('*summary*.csv'))
    
    print(f"--- Analizando nombres de métricas en {len(summary_files)} archivos ---")

    for file_path in summary_files:
        try:
            df = pd.read_csv(file_path)
            if 'Métrica' in df.columns:
                all_metrics.update(df['Métrica'].unique())
            else:
                print(f"Advertencia: El archivo {file_path} no tiene la columna 'Métrica'.")
        except Exception as e:
            print(f"Error al procesar {file_path}: {e}")
    
    print("\n--- Nombres de Métricas Únicos Encontrados ---")
    for metric in sorted(list(all_metrics)):
        print(metric)
    print("\n--- Fin del Diagnóstico ---")

if __name__ == "__main__":
    diagnose_metric_names(RESULTS_DIR)
