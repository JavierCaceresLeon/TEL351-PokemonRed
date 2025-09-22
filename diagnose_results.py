
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path('RESULTADOS')

def diagnose_summary_files(results_dir):
    """
    Escanea todos los archivos *summary*.csv, imprime sus columnas
    y reporta cualquier inconsistencia.
    """
    summary_files = list(results_dir.rglob('*summary*.csv'))
    
    if not summary_files:
        print("No se encontraron archivos de resumen para diagnosticar.")
        return

    print(f"--- Diagnóstico de {len(summary_files)} archivos de resumen ---")
    
    reference_columns = None
    inconsistent_files = []

    for i, file_path in enumerate(summary_files):
        try:
            df = pd.read_csv(file_path)
            columns = set(df.columns)
            
            print(f"\n{i+1}. Archivo: {file_path.relative_to(results_dir)}")
            print(f"   Columnas: {sorted(list(columns))}")

            if reference_columns is None:
                reference_columns = columns
            
            if columns != reference_columns:
                inconsistent_files.append({
                    "path": file_path,
                    "columns": columns,
                    "missing": reference_columns - columns,
                    "extra": columns - reference_columns
                })

        except Exception as e:
            print(f"Error al procesar {file_path}: {e}")

    if inconsistent_files:
        print("\n\n--- RESUMEN DE INCONSISTENCIAS ---")
        print(f"Columnas de referencia (del primer archivo): {sorted(list(reference_columns))}")
        for issue in inconsistent_files:
            print(f"\nArchivo inconsistente: {issue['path'].relative_to(results_dir)}")
            if issue['missing']:
                print(f"  - Faltan columnas: {sorted(list(issue['missing']))}")
            if issue['extra']:
                print(f"  - Columnas extra: {sorted(list(issue['extra']))}")
    else:
        print("\n\n--- CONCLUSIÓN ---")
        print("Todos los archivos de resumen parecen tener columnas consistentes.")

if __name__ == '__main__':
    diagnose_summary_files(RESULTS_DIR)
