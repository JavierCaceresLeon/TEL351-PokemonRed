"""
Script para visualizar y analizar los resultados de la comparación
Genera gráficos y estadísticas detalladas
"""

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

class ResultsAnalyzer:
    """Analizador de resultados de comparación"""
    
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.data = None
        
    def load_latest_results(self):
        """Cargar los resultados más recientes"""
        if not self.results_dir.exists():
            print("Directorio de resultados no encontrado")
            return False
        
        # Buscar archivos de resultados
        result_files = list(self.results_dir.glob("comparison_results_*.json"))
        
        if not result_files:
            print("No se encontraron archivos de resultados")
            return False
        
        # Obtener el más reciente
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        print(f"Cargando resultados de: {latest_file.name}")
        
        with open(latest_file, 'r') as f:
            raw_data = json.load(f)
        
        # Convertir a DataFrame
        all_results = []
        for agent_type, runs in raw_data.items():
            for run in runs:
                run['agent_type'] = agent_type
                all_results.append(run)
        
        self.data = pd.DataFrame(all_results)
        return True
    
    def print_summary(self):
        """Imprimir resumen de resultados"""
        if self.data is None:
            print("No hay datos cargados")
            return
        
        print("\n=== RESUMEN DE RESULTADOS ===")
        
        # Resumen por agente
        summary = self.data.groupby('agent_type').agg({
            'success': ['count', 'sum', 'mean'],
            'total_time': ['mean', 'std', 'min', 'max'],
            'plan_length': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        print("\nEstadísticas por Agente:")
        print(summary)
        
        # Tasa de éxito
        print("\n--- Tasas de Éxito ---")
        success_rates = self.data.groupby('agent_type')['success'].agg(['count', 'sum']).eval('success_rate = sum / count * 100')
        for agent, row in success_rates.iterrows():
            print(f"{agent}: {row['success_rate']:.1f}% ({row['sum']}/{row['count']})")
        
        # Mejores tiempos
        print("\n--- Mejores Tiempos (solo éxitos) ---")
        successful = self.data[self.data['success'] == True]
        if not successful.empty:
            best_times = successful.groupby('agent_type')['total_time'].min()
            for agent, time in best_times.items():
                print(f"{agent}: {time:.2f}s")
        
    def create_visualizations(self):
        """Crear visualizaciones de los resultados"""
        if self.data is None:
            print("No hay datos para visualizar")
            return
        
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Comparación de Agentes - Pokémon Red', fontsize=16)
        
        # 1. Tasa de éxito por agente
        success_data = self.data.groupby('agent_type')['success'].agg(['count', 'sum']).eval('success_rate = sum / count * 100')
        axes[0, 0].bar(success_data.index, success_data['success_rate'])
        axes[0, 0].set_title('Tasa de Éxito por Agente')
        axes[0, 0].set_ylabel('Porcentaje de Éxito (%)')
        axes[0, 0].set_ylim(0, 100)
        
        # Agregar valores en las barras
        for i, v in enumerate(success_data['success_rate']):
            axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center')
        
        # 2. Distribución de tiempos (solo éxitos)
        successful = self.data[self.data['success'] == True]
        if not successful.empty:
            sns.boxplot(data=successful, x='agent_type', y='total_time', ax=axes[0, 1])
            axes[0, 1].set_title('Distribución de Tiempos (Solo Éxitos)')
            axes[0, 1].set_ylabel('Tiempo Total (s)')
        else:
            axes[0, 1].text(0.5, 0.5, 'No hay éxitos para mostrar', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Distribución de Tiempos')
        
        # 3. Número de pasos por agente
        if not successful.empty:
            sns.violinplot(data=successful, x='agent_type', y='plan_length', ax=axes[1, 0])
            axes[1, 0].set_title('Distribución de Pasos (Solo Éxitos)')
            axes[1, 0].set_ylabel('Número de Pasos')
        else:
            axes[1, 0].text(0.5, 0.5, 'No hay éxitos para mostrar', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Distribución de Pasos')
        
        # 4. Comparación tiempo vs pasos
        if not successful.empty:
            for agent in successful['agent_type'].unique():
                agent_data = successful[successful['agent_type'] == agent]
                axes[1, 1].scatter(agent_data['plan_length'], agent_data['total_time'], 
                                 label=agent, alpha=0.7, s=60)
            
            axes[1, 1].set_xlabel('Número de Pasos')
            axes[1, 1].set_ylabel('Tiempo Total (s)')
            axes[1, 1].set_title('Eficiencia: Tiempo vs Pasos')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No hay éxitos para mostrar', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Eficiencia: Tiempo vs Pasos')
        
        plt.tight_layout()
        
        # Guardar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"analysis_plot_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nGráfico guardado en: {plot_file}")
        
        plt.show()
    
    def detailed_analysis(self):
        """Análisis detallado de los resultados"""
        if self.data is None:
            print("No hay datos para analizar")
            return
        
        print("\n=== ANÁLISIS DETALLADO ===")
        
        # Análisis de correlaciones
        numeric_cols = ['total_time', 'plan_length']
        if 'nodes_explored' in self.data.columns:
            numeric_cols.append('nodes_explored')
        if 'iterations' in self.data.columns:
            numeric_cols.append('iterations')
        
        successful = self.data[self.data['success'] == True]
        
        if not successful.empty and len(numeric_cols) > 1:
            print("\n--- Correlaciones (Solo Éxitos) ---")
            corr_matrix = successful[numeric_cols].corr()
            print(corr_matrix.round(3))
        
        # Análisis por agente
        for agent in self.data['agent_type'].unique():
            agent_data = self.data[self.data['agent_type'] == agent]
            agent_success = agent_data[agent_data['success'] == True]
            
            print(f"\n--- Análisis Detallado: {agent.upper()} ---")
            print(f"Intentos totales: {len(agent_data)}")
            print(f"Éxitos: {len(agent_success)}")
            
            if len(agent_success) > 0:
                print(f"Tiempo promedio: {agent_success['total_time'].mean():.2f}s ± {agent_success['total_time'].std():.2f}s")
                print(f"Pasos promedio: {agent_success['plan_length'].mean():.1f} ± {agent_success['plan_length'].std():.1f}")
                print(f"Mejor tiempo: {agent_success['total_time'].min():.2f}s")
                print(f"Menos pasos: {agent_success['plan_length'].min()}")
        
        # Recomendaciones
        print("\n=== RECOMENDACIONES ===")
        
        if successful.empty:
            print("⚠️ Ningún agente logró completar el objetivo.")
            print("Sugerencias:")
            print("- Aumentar max_steps en la configuración")
            print("- Ajustar parámetros de búsqueda")
            print("- Verificar heurísticas y funciones de fitness")
        else:
            # Encontrar el mejor agente
            avg_times = successful.groupby('agent_type')['total_time'].mean()
            best_agent = avg_times.idxmin()
            
            print(f"🏆 Mejor agente por tiempo promedio: {best_agent}")
            print(f"   Tiempo promedio: {avg_times[best_agent]:.2f}s")
            
            # Tasa de éxito
            success_rates = self.data.groupby('agent_type')['success'].mean()
            most_reliable = success_rates.idxmax()
            
            print(f"🎯 Agente más confiable: {most_reliable}")
            print(f"   Tasa de éxito: {success_rates[most_reliable]*100:.1f}%")

def main():
    """Función principal"""
    print("Analizador de Resultados - Comparación de Agentes")
    print("=" * 50)
    
    analyzer = ResultsAnalyzer()
    
    if not analyzer.load_latest_results():
        print("No se pudieron cargar los resultados.")
        print("Asegúrate de haber ejecutado run_comparison.py primero.")
        return
    
    # Análisis básico
    analyzer.print_summary()
    
    # Análisis detallado
    analyzer.detailed_analysis()
    
    # Crear visualizaciones
    try:
        analyzer.create_visualizations()
    except ImportError:
        print("\n⚠️ matplotlib/seaborn no disponible para visualizaciones")
    except Exception as e:
        print(f"\n⚠️ Error creando visualizaciones: {e}")

if __name__ == "__main__":
    main()
