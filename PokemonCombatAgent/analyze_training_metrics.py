"""
Visualizar métricas de entrenamiento desde logs de TensorBoard
"""
import argparse
from pathlib import Path
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import json

def extract_tensorboard_metrics(logdir):
    """Extrae métricas de archivos TensorBoard"""
    ea = event_accumulator.EventAccumulator(str(logdir))
    ea.Reload()
    
    metrics = {}
    
    # Obtener todas las métricas disponibles
    for tag in ea.Tags()['scalars']:
        try:
            events = ea.Scalars(tag)
            metrics[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events],
                'wall_time': [e.wall_time for e in events]
            }
        except:
            pass
    
    return metrics

def plot_training_metrics(metrics, output_dir):
    """Genera gráficos de las métricas principales"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Métricas importantes para combate
    key_metrics = {
        'rollout/ep_rew_mean': 'Recompensa Promedio por Episodio',
        'rollout/ep_len_mean': 'Longitud Promedio de Episodio',
        'train/learning_rate': 'Learning Rate',
        'train/explained_variance': 'Explained Variance',
        'train/approx_kl': 'Approximate KL Divergence',
        'train/value_loss': 'Value Loss',
        'train/policy_gradient_loss': 'Policy Gradient Loss',
        'train/entropy_loss': 'Entropy Loss',
    }
    
    # Crear gráfico por cada métrica
    for metric_key, title in key_metrics.items():
        if metric_key in metrics:
            plt.figure(figsize=(10, 6))
            data = metrics[metric_key]
            plt.plot(data['steps'], data['values'], linewidth=2)
            plt.title(title)
            plt.xlabel('Timesteps')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = metric_key.replace('/', '_') + '.png'
            plt.savefig(output_dir / filename, dpi=150)
            plt.close()
            print(f"  📊 {filename}")
    
    # Crear gráfico combinado de métricas de entrenamiento
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Resumen de Métricas de Entrenamiento', fontsize=16)
    
    plots = [
        ('rollout/ep_rew_mean', 'Recompensa Promedio', axes[0, 0]),
        ('train/explained_variance', 'Explained Variance', axes[0, 1]),
        ('train/approx_kl', 'KL Divergence', axes[1, 0]),
        ('train/value_loss', 'Value Loss', axes[1, 1])
    ]
    
    for metric_key, title, ax in plots:
        if metric_key in metrics:
            data = metrics[metric_key]
            ax.plot(data['steps'], data['values'], linewidth=2)
            ax.set_title(title)
            ax.set_xlabel('Timesteps')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_summary.png', dpi=150)
    plt.close()
    print(f"  📊 training_summary.png")

def export_metrics_csv(metrics, output_file):
    """Exporta métricas a CSV"""
    # Encontrar la longitud máxima
    max_len = max(len(m['steps']) for m in metrics.values())
    
    # Crear DataFrame
    data = {}
    for metric_name, metric_data in metrics.items():
        steps = metric_data['steps']
        values = metric_data['values']
        
        # Rellenar con NaN si es más corto
        if len(steps) < max_len:
            steps = steps + [None] * (max_len - len(steps))
            values = values + [None] * (max_len - len(values))
        
        data[f"{metric_name}_step"] = steps
        data[f"{metric_name}_value"] = values
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"  💾 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analizar métricas de entrenamiento')
    parser.add_argument('--session-dir', required=True, help='Directorio de la sesión (ej: sessions/combat_agent_final_battle_loop)')
    parser.add_argument('--output-dir', default='training_analysis', help='Directorio de salida')
    args = parser.parse_args()
    
    session_dir = Path(args.session_dir)
    
    # Buscar archivos de TensorBoard
    tb_dirs = list(session_dir.glob('**/events.out.tfevents.*'))
    
    if not tb_dirs:
        print(f"❌ No se encontraron archivos TensorBoard en {session_dir}")
        print("   Buscando en subdirectorios...")
        
        # Buscar en PPO_1, PPO_2, etc.
        for subdir in session_dir.glob('PPO_*'):
            tb_files = list(subdir.glob('**/events.out.tfevents.*'))
            if tb_files:
                tb_dirs.extend(tb_files)
    
    if not tb_dirs:
        print("❌ No se encontraron logs de TensorBoard")
        return
    
    print(f"\n📂 Encontrados {len(tb_dirs)} archivos de TensorBoard")
    
    # Procesar todos los logs
    all_metrics = {}
    for tb_file in tb_dirs:
        print(f"\n📖 Procesando: {tb_file.parent}")
        metrics = extract_tensorboard_metrics(tb_file.parent)
        
        # Combinar métricas
        for key, value in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = value
            else:
                # Agregar datos
                all_metrics[key]['steps'].extend(value['steps'])
                all_metrics[key]['values'].extend(value['values'])
                all_metrics[key]['wall_time'].extend(value['wall_time'])
    
    if not all_metrics:
        print("❌ No se pudieron extraer métricas")
        return
    
    print(f"\n📊 Métricas disponibles:")
    for metric_name in sorted(all_metrics.keys()):
        count = len(all_metrics[metric_name]['steps'])
        print(f"  • {metric_name} ({count} puntos)")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generar visualizaciones
    print(f"\n📈 Generando gráficos en {output_dir}/...")
    plot_training_metrics(all_metrics, output_dir)
    
    # Exportar CSV
    csv_file = output_dir / 'metrics.csv'
    print(f"\n💾 Exportando métricas a CSV...")
    export_metrics_csv(all_metrics, csv_file)
    
    # Guardar resumen JSON
    summary = {
        'total_metrics': len(all_metrics),
        'metrics': {
            name: {
                'count': len(data['steps']),
                'last_value': data['values'][-1] if data['values'] else None,
                'last_step': data['steps'][-1] if data['steps'] else None
            }
            for name, data in all_metrics.items()
        }
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Análisis completado!")
    print(f"   Resultados en: {output_dir}/")
    print(f"   • Gráficos PNG")
    print(f"   • metrics.csv")
    print(f"   • summary.json")

if __name__ == '__main__':
    main()
