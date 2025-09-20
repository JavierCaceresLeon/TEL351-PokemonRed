"""
Generador de Visualizaciones para Métricas de Agentes
====================================================

Este script genera gráficos y visualizaciones comparativas de las métricas
capturadas por los agentes Epsilon Greedy, PPO y Tabu Search.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob
from datetime import datetime

class MetricsVisualizer:
    def __init__(self):
        self.epsilon_results_dir = Path("results")
        self.ppo_results_dir = Path("../v2/ppo_results")
        self.tabu_results_dir = Path("results")  # Tabu Search usa el mismo directorio
        self.output_dir = Path("visualization_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Configurar estilo de gráficos
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_all_metrics(self):
        """Cargar todas las métricas disponibles de los tres agentes"""
        epsilon_data = []
        ppo_data = []
        tabu_data = []
        
        # Cargar datos de Epsilon Greedy
        if self.epsilon_results_dir.exists():
            epsilon_json_files = glob.glob(str(self.epsilon_results_dir / "epsilon_greedy_raw_data_*.json"))
            for file_path in epsilon_json_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data['agent_type'] = 'Epsilon Greedy'
                        data['file_path'] = file_path
                        epsilon_data.append(data)
                except Exception as e:
                    print(f"Error cargando {file_path}: {e}")
        
        # Cargar datos de PPO
        if self.ppo_results_dir.exists():
            ppo_json_files = glob.glob(str(self.ppo_results_dir / "ppo_raw_data_*.json"))
            for file_path in ppo_json_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data['agent_type'] = 'PPO'
                        data['file_path'] = file_path
                        ppo_data.append(data)
                except Exception as e:
                    print(f"Error cargando {file_path}: {e}")
        
        # Cargar datos de Tabu Search
        if self.tabu_results_dir.exists():
            tabu_json_files = glob.glob(str(self.tabu_results_dir / "tabu_search_raw_data_*.json"))
            for file_path in tabu_json_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data['agent_type'] = 'Tabu Search'
                        data['file_path'] = file_path
                        tabu_data.append(data)
                except Exception as e:
                    print(f"Error cargando {file_path}: {e}")
        
        return epsilon_data, ppo_data, tabu_data
    
    def create_performance_comparison(self, epsilon_data, ppo_data, tabu_data):
        """Crear gráfico de comparación de rendimiento entre los tres agentes"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparación de Rendimiento: Epsilon Greedy vs PPO vs Tabu Search', 
                     fontsize=16, fontweight='bold')
        
        # Preparar datos para comparación
        comparison_data = []
        
        for data in epsilon_data:
            performance = data.get('game_performance', {})
            session = data.get('session_info', {})
            comparison_data.append({
                'Agent': 'Epsilon Greedy',
                'Total_Reward': performance.get('total_reward', 0),
                'Total_Steps': session.get('total_steps', 0),
                'Elapsed_Time': session.get('duration_seconds', 0),
                'Steps_Per_Second': performance.get('steps_per_second', 0),
                'Avg_Reward_Per_Step': performance.get('average_reward_per_step', 0)
            })
        
        for data in ppo_data:
            performance = data.get('game_performance', {})
            session = data.get('session_info', {})
            comparison_data.append({
                'Agent': 'PPO',
                'Total_Reward': performance.get('total_reward', 0),
                'Total_Steps': session.get('total_steps', 0),
                'Elapsed_Time': session.get('duration_seconds', 0),
                'Steps_Per_Second': performance.get('steps_per_second', 0),
                'Avg_Reward_Per_Step': performance.get('average_reward_per_step', 0)
            })
        
        for data in tabu_data:
            performance = data.get('game_performance', {})
            session = data.get('session_info', {})
            comparison_data.append({
                'Agent': 'Tabu Search',
                'Total_Reward': performance.get('total_reward', 0),
                'Total_Steps': session.get('total_steps', 0),
                'Elapsed_Time': session.get('duration_seconds', 0),
                'Steps_Per_Second': performance.get('steps_per_second', 0),
                'Avg_Reward_Per_Step': performance.get('average_reward_per_step', 0)
            })
        
        if not comparison_data:
            print("No hay datos para visualizar")
            return
        
        df = pd.DataFrame(comparison_data)
        
        # Gráfico 1: Recompensa Total
        sns.boxplot(data=df, x='Agent', y='Total_Reward', ax=axes[0,0])
        axes[0,0].set_title('Distribución de Recompensa Total')
        axes[0,0].set_ylabel('Recompensa Total')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Gráfico 2: Pasos por Segundo
        sns.boxplot(data=df, x='Agent', y='Steps_Per_Second', ax=axes[0,1])
        axes[0,1].set_title('Velocidad de Ejecución (Pasos/Segundo)')
        axes[0,1].set_ylabel('Pasos por Segundo')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Gráfico 3: Recompensa Promedio por Paso
        sns.boxplot(data=df, x='Agent', y='Avg_Reward_Per_Step', ax=axes[1,0])
        axes[1,0].set_title('Eficiencia (Recompensa/Paso)')
        axes[1,0].set_ylabel('Recompensa Promedio por Paso')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Gráfico 4: Tiempo Total
        sns.boxplot(data=df, x='Agent', y='Elapsed_Time', ax=axes[1,1])
        axes[1,1].set_title('Duración de Sesiones')
        axes[1,1].set_ylabel('Tiempo Transcurrido (s)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = self.output_dir / f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Gráfico guardado: {output_path}")
        
        return df
    
    def create_action_distribution_comparison(self, epsilon_data, ppo_data, tabu_data):
        """Crear comparación de distribución de acciones entre los tres agentes"""
        action_names = ['↓ (DOWN)', '← (LEFT)', '→ (RIGHT)', '↑ (UP)', 'A', 'B', 'START']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Distribución de Acciones por Agente', fontsize=16, fontweight='bold')
        
        # Epsilon Greedy
        if epsilon_data:
            epsilon_actions = []
            for data in epsilon_data:
                raw_data = data.get('raw_data', {})
                epsilon_actions.extend(raw_data.get('action_history', []))
            
            if epsilon_actions:
                action_counts = [epsilon_actions.count(i) for i in range(7)]
                action_percentages = [count/sum(action_counts)*100 for count in action_counts]
                
                axes[0].pie(action_percentages, labels=action_names, autopct='%1.1f%%', startangle=90)
                axes[0].set_title(f'Epsilon Greedy\n({len(epsilon_actions):,} acciones totales)')
        
        # PPO
        if ppo_data:
            ppo_actions = []
            for data in ppo_data:
                raw_data = data.get('raw_data', {})
                ppo_actions.extend(raw_data.get('action_history', []))
            
            if ppo_actions:
                action_counts = [ppo_actions.count(i) for i in range(7)]
                action_percentages = [count/sum(action_counts)*100 for count in action_counts]
                
                axes[1].pie(action_percentages, labels=action_names, autopct='%1.1f%%', startangle=90)
                axes[1].set_title(f'PPO\n({len(ppo_actions):,} acciones totales)')
        
        # Tabu Search
        if tabu_data:
            tabu_actions = []
            for data in tabu_data:
                raw_data = data.get('raw_data', {})
                tabu_actions.extend(raw_data.get('action_history', []))
            
            if tabu_actions:
                action_counts = [tabu_actions.count(i) for i in range(7)]
                action_percentages = [count/sum(action_counts)*100 for count in action_counts]
                
                axes[2].pie(action_percentages, labels=action_names, autopct='%1.1f%%', startangle=90)
                axes[2].set_title(f'Tabu Search\n({len(tabu_actions):,} acciones totales)')
        
        plt.tight_layout()
        output_path = self.output_dir / f"action_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Gráfico guardado: {output_path}")
    
    def create_reward_progression(self, epsilon_data, ppo_data, tabu_data):
        """Crear gráfico de progresión de recompensas"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Progresión de Recompensas Durante las Sesiones', fontsize=16, fontweight='bold')
        
        # Epsilon Greedy
        for i, data in enumerate(epsilon_data[:3]):  # Máximo 3 sesiones para claridad
            raw_data = data.get('raw_data', {})
            rewards = raw_data.get('reward_history', [])
            if rewards:
                cumulative_rewards = np.cumsum(rewards)
                axes[0].plot(cumulative_rewards, label=f'Sesión {i+1}', alpha=0.7, linewidth=2)
        
        axes[0].set_title('Epsilon Greedy - Recompensa Acumulada')
        axes[0].set_xlabel('Pasos')
        axes[0].set_ylabel('Recompensa Acumulada')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # PPO
        for i, data in enumerate(ppo_data[:3]):  # Máximo 3 sesiones para claridad
            raw_data = data.get('raw_data', {})
            rewards = raw_data.get('reward_history', [])
            if rewards:
                cumulative_rewards = np.cumsum(rewards)
                axes[1].plot(cumulative_rewards, label=f'Sesión {i+1}', alpha=0.7, linewidth=2)
        
        axes[1].set_title('PPO - Recompensa Acumulada')
        axes[1].set_xlabel('Pasos')
        axes[1].set_ylabel('Recompensa Acumulada')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Tabu Search
        for i, data in enumerate(tabu_data[:3]):  # Máximo 3 sesiones para claridad
            raw_data = data.get('raw_data', {})
            rewards = raw_data.get('reward_history', [])
            if rewards:
                cumulative_rewards = np.cumsum(rewards)
                axes[2].plot(cumulative_rewards, label=f'Sesión {i+1}', alpha=0.7, linewidth=2)
        
        axes[2].set_title('Tabu Search - Recompensa Acumulada')
        axes[2].set_xlabel('Pasos')
        axes[2].set_ylabel('Recompensa Acumulada')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f"reward_progression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Gráfico guardado: {output_path}")
    
    def create_resource_usage_comparison(self, epsilon_data, ppo_data):
        """Crear comparación de uso de recursos"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Uso de Recursos del Sistema', fontsize=16, fontweight='bold')
        
        # Preparar datos de memoria
        memory_data = []
        
        for data in epsilon_data:
            memory_data.append({
                'Agent': 'Epsilon Greedy',
                'Memory_MB': data['system_resources']['memory_mb']
            })
        
        for data in ppo_data:
            memory_data.append({
                'Agent': 'PPO',
                'Memory_MB': data['system_resources']['memory_mb']
            })
        
        if memory_data:
            df_memory = pd.DataFrame(memory_data)
            
            # Gráfico de memoria
            sns.boxplot(data=df_memory, x='Agent', y='Memory_MB', ax=axes[0])
            axes[0].set_title('Uso de Memoria RAM')
            axes[0].set_ylabel('Memoria (MB)')
            
            # Gráfico de barras comparativo
            memory_avg = df_memory.groupby('Agent')['Memory_MB'].mean()
            memory_avg.plot(kind='bar', ax=axes[1], color=['skyblue', 'lightcoral'])
            axes[1].set_title('Memoria Promedio por Agente')
            axes[1].set_ylabel('Memoria Promedio (MB)')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = self.output_dir / f"resource_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Gráfico guardado: {output_path}")
    
    def generate_summary_report(self, epsilon_data, ppo_data):
        """Generar reporte resumen en markdown"""
        report_path = self.output_dir / f"metrics_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"""# 📊 Reporte de Análisis Comparativo
## Epsilon Greedy vs PPO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### 📈 **Resumen de Datos**
- **Sesiones Epsilon Greedy:** {len(epsilon_data)}
- **Sesiones PPO:** {len(ppo_data)}
- **Total de Sesiones:** {len(epsilon_data) + len(ppo_data)}

### 🎯 **Estadísticas de Rendimiento**

#### Epsilon Greedy
""")
            
            if epsilon_data:
                total_rewards = [d['session_info']['total_reward'] for d in epsilon_data]
                total_steps = [d['session_info']['total_steps'] for d in epsilon_data]
                f.write(f"""- **Recompensa Promedio:** {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}
- **Recompensa Máxima:** {np.max(total_rewards):.2f}
- **Recompensa Mínima:** {np.min(total_rewards):.2f}
- **Pasos Promedio:** {np.mean(total_steps):,.0f} ± {np.std(total_steps):,.0f}
""")
            
            f.write(f"""
#### PPO
""")
            
            if ppo_data:
                total_rewards = [d['session_info']['total_reward'] for d in ppo_data]
                total_steps = [d['session_info']['total_steps'] for d in ppo_data]
                f.write(f"""- **Recompensa Promedio:** {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}
- **Recompensa Máxima:** {np.max(total_rewards):.2f}
- **Recompensa Mínima:** {np.min(total_rewards):.2f}
- **Pasos Promedio:** {np.mean(total_steps):,.0f} ± {np.std(total_steps):,.0f}
""")
            
            f.write(f"""
### 🎮 **Observaciones**
- Los gráficos generados están disponibles en la carpeta `{self.output_dir}`
- Datos crudos disponibles en formato JSON en las carpetas respectivas
- Este análisis se actualizará automáticamente con nuevas sesiones

### 📁 **Archivos Generados**
- Comparación de rendimiento
- Distribución de acciones
- Progresión de recompensas
- Uso de recursos del sistema

---
*Reporte generado automáticamente por el sistema de métricas avanzadas*
""")
        
        print(f"📄 Reporte guardado: {report_path}")
        return report_path
    
    def generate_all_visualizations(self):
        """Generar todas las visualizaciones disponibles"""
        print("🔄 Cargando datos de métricas...")
        epsilon_data, ppo_data, tabu_data = self.load_all_metrics()
        
        print(f"📊 Datos encontrados:")
        print(f"   - Epsilon Greedy: {len(epsilon_data)} sesiones")
        print(f"   - PPO: {len(ppo_data)} sesiones")
        print(f"   - Tabu Search: {len(tabu_data)} sesiones")
        
        if not epsilon_data and not ppo_data and not tabu_data:
            print("❌ No se encontraron datos para visualizar")
            return
        
        print("\n🎨 Generando visualizaciones...")
        
        # Generar todos los gráficos
        if epsilon_data or ppo_data or tabu_data:
            self.create_performance_comparison(epsilon_data, ppo_data, tabu_data)
            self.create_action_distribution_comparison(epsilon_data, ppo_data, tabu_data)
            self.create_reward_progression(epsilon_data, ppo_data, tabu_data)
            # self.create_resource_usage_comparison(epsilon_data, ppo_data, tabu_data)  # TODO: Update this method
            # self.generate_summary_report(epsilon_data, ppo_data, tabu_data)  # TODO: Update this method
        
        print(f"\n✅ Visualizaciones completadas. Archivos en: {self.output_dir}")

if __name__ == "__main__":
    print("🎨 Generador de Visualizaciones de Métricas")
    print("=" * 50)
    
    visualizer = MetricsVisualizer()
    visualizer.generate_all_visualizations()