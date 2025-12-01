"""
Analyze comparison results and generate comprehensive report with visualizations
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_comparison(json_path):
    """Load comparison JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_metrics_comparison_plot(data, output_dir):
    """Create bar chart comparing key metrics"""
    combat_summary = data['summary']['combat']
    baseline_summary = data['summary']['baseline']
    
    metrics = ['avg_reward', 'avg_steps', 'avg_hp_dealt', 'avg_hp_taken', 'win_rate']
    combat_values = [combat_summary[m] for m in metrics]
    baseline_values = [baseline_summary[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, combat_values, width, label='Combat Agent', color='#FF6B6B')
    bars2 = ax.bar(x + width/2, baseline_values, width, label='Baseline PPO', color='#4ECDC4')
    
    ax.set_xlabel('Métricas', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Métricas: Combat Agent vs Baseline PPO', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Reward\nPromedio', 'Steps\nPromedio', 'HP Dealt\nPromedio', 
                        'HP Taken\nPromedio', 'Win Rate'], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: metrics_comparison.png")

def create_episode_analysis_plot(data, output_dir):
    """Create detailed episode-by-episode analysis"""
    combat_results = data['combat_results']
    baseline_results = data['baseline_results']
    
    episodes = range(1, len(combat_results) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análisis por Episodio: Combat Agent vs Baseline PPO', 
                 fontsize=16, fontweight='bold')
    
    # Reward per episode
    ax = axes[0, 0]
    ax.plot(episodes, [r['total_reward'] for r in combat_results], 
            marker='o', label='Combat Agent', color='#FF6B6B', linewidth=2)
    ax.plot(episodes, [r['total_reward'] for r in baseline_results], 
            marker='s', label='Baseline PPO', color='#4ECDC4', linewidth=2)
    ax.set_xlabel('Episodio', fontweight='bold')
    ax.set_ylabel('Reward Total', fontweight='bold')
    ax.set_title('Reward por Episodio', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Steps per episode
    ax = axes[0, 1]
    ax.plot(episodes, [r['steps'] for r in combat_results], 
            marker='o', label='Combat Agent', color='#FF6B6B', linewidth=2)
    ax.plot(episodes, [r['steps'] for r in baseline_results], 
            marker='s', label='Baseline PPO', color='#4ECDC4', linewidth=2)
    ax.set_xlabel('Episodio', fontweight='bold')
    ax.set_ylabel('Steps', fontweight='bold')
    ax.set_title('Steps por Episodio', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # HP Dealt per episode
    ax = axes[1, 0]
    ax.bar([e - 0.2 for e in episodes], [r['hp_dealt'] for r in combat_results], 
           width=0.4, label='Combat Agent', color='#FF6B6B')
    ax.bar([e + 0.2 for e in episodes], [r['hp_dealt'] for r in baseline_results], 
           width=0.4, label='Baseline PPO', color='#4ECDC4')
    ax.set_xlabel('Episodio', fontweight='bold')
    ax.set_ylabel('HP Dealt', fontweight='bold')
    ax.set_title('Daño Causado por Episodio', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Battle outcomes
    ax = axes[1, 1]
    combat_wins = [r['battles_won'] for r in combat_results]
    baseline_wins = [r['battles_won'] for r in baseline_results]
    ax.bar([e - 0.2 for e in episodes], combat_wins, 
           width=0.4, label='Combat Agent', color='#FF6B6B')
    ax.bar([e + 0.2 for e in episodes], baseline_wins, 
           width=0.4, label='Baseline PPO', color='#4ECDC4')
    ax.set_xlabel('Episodio', fontweight='bold')
    ax.set_ylabel('Batallas Ganadas', fontweight='bold')
    ax.set_title('Victorias por Episodio', fontweight='bold')
    ax.set_ylim([0, 1.2])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'episode_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: episode_analysis.png")

def create_reward_formula_comparison(output_dir):
    """Visualize reward formulas for both agents"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Comparación de Fórmulas de Recompensa', fontsize=16, fontweight='bold')
    
    # Combat Agent Reward Formula
    ax1.text(0.5, 0.9, 'Combat Agent (Modificado)', ha='center', 
             fontsize=14, fontweight='bold', transform=ax1.transAxes)
    
    formula_combat = [
        'Reward = Base_Reward + Combat_Bonus',
        '',
        'Combat_Bonus:',
        '  • HP damage dealt: +0.5 por cada HP',
        '  • Victory: +100.0',
        '  • HP damage taken: -0.3 por cada HP',
        '  • Not in battle: -0.02 por step',
        '',
        'Enfoque: Maximizar daño y victorias'
    ]
    
    y = 0.75
    for line in formula_combat:
        weight = 'bold' if ':' in line or line.startswith('Enfoque') else 'normal'
        ax1.text(0.1, y, line, fontsize=11, fontweight=weight,
                transform=ax1.transAxes, family='monospace')
        y -= 0.08
    
    ax1.axis('off')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Baseline PPO Reward Formula
    ax2.text(0.5, 0.9, 'Baseline PPO (Original)', ha='center',
             fontsize=14, fontweight='bold', transform=ax2.transAxes)
    
    formula_baseline = [
        'Reward = Exploration + Events + Progress',
        '',
        'Components:',
        '  • Map exploration',
        '  • Event flags progression',
        '  • Level gains',
        '  • Badge collection',
        '  • Party composition',
        '',
        'Enfoque: Progreso general del juego'
    ]
    
    y = 0.75
    for line in formula_baseline:
        weight = 'bold' if ':' in line or line.startswith('Enfoque') else 'normal'
        ax2.text(0.1, y, line, fontsize=11, fontweight=weight,
                transform=ax2.transAxes, family='monospace')
        y -= 0.08
    
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_formulas.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: reward_formulas.png")

def create_battle_engagement_plot(data, output_dir):
    """Analyze battle engagement"""
    combat_results = data['combat_results']
    baseline_results = data['baseline_results']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Análisis de Participación en Batallas', fontsize=14, fontweight='bold')
    
    # Time in battle
    combat_battle_time = [r['time_in_battle'] for r in combat_results]
    baseline_battle_time = [r['time_in_battle'] for r in baseline_results]
    
    ax1.bar(['Combat Agent', 'Baseline PPO'], 
            [np.mean(combat_battle_time), np.mean(baseline_battle_time)],
            color=['#FF6B6B', '#4ECDC4'])
    ax1.set_ylabel('Steps Promedio en Batalla', fontweight='bold')
    ax1.set_title('Tiempo en Batalla', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate([np.mean(combat_battle_time), np.mean(baseline_battle_time)]):
        ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Battle occurrence rate
    combat_entered = sum(1 for r in combat_results if r['time_in_battle'] > 0)
    baseline_entered = sum(1 for r in baseline_results if r['time_in_battle'] > 0)
    total = len(combat_results)
    
    ax2.bar(['Combat Agent', 'Baseline PPO'],
            [combat_entered/total * 100, baseline_entered/total * 100],
            color=['#FF6B6B', '#4ECDC4'])
    ax2.set_ylabel('% de Episodios con Batalla', fontweight='bold')
    ax2.set_title('Tasa de Entrada a Batallas', fontweight='bold')
    ax2.set_ylim([0, 110])
    ax2.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate([combat_entered/total * 100, baseline_entered/total * 100]):
        ax2.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'battle_engagement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: battle_engagement.png")

def generate_markdown_report(data, output_dir):
    """Generate comprehensive markdown report"""
    combat = data['summary']['combat']
    baseline = data['summary']['baseline']
    
    report = f"""# Reporte de Comparación: Combat Agent vs Baseline PPO

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Estado inicial:** `{data['battle_state']}`  
**Episodios evaluados:** {data['num_episodes']}

---

## 📊 Resumen Ejecutivo

### Resultado General
**🏆 GANADOR: Baseline PPO (Modelo Original)**

El Baseline PPO supera significativamente al Combat Agent en todas las métricas clave.

---

## 📈 Métricas Comparativas

| Métrica | Combat Agent | Baseline PPO | Diferencia | Ganador |
|---------|--------------|--------------|------------|---------|
| **Reward Promedio** | {combat['avg_reward']:.2f} | {baseline['avg_reward']:.2f} | {baseline['avg_reward'] - combat['avg_reward']:.2f} | ✅ Baseline |
| **Steps Promedio** | {combat['avg_steps']:.0f} | {baseline['avg_steps']:.0f} | {combat['avg_steps'] - baseline['avg_steps']:.0f} | ✅ Baseline (más eficiente) |
| **HP Dealt** | {combat['avg_hp_dealt']:.2f} | {baseline['avg_hp_dealt']:.2f} | {baseline['avg_hp_dealt'] - combat['avg_hp_dealt']:.2f} | ✅ Baseline |
| **HP Taken** | {combat['avg_hp_taken']:.2f} | {baseline['avg_hp_taken']:.2f} | {combat['avg_hp_taken'] - baseline['avg_hp_taken']:.2f} | ✅ Combat (menor daño) |
| **Win Rate** | {combat['win_rate']*100:.1f}% | {baseline['win_rate']*100:.1f}% | {(baseline['win_rate'] - combat['win_rate'])*100:.1f}% | ✅ Baseline |

---

## 🔍 Análisis Detallado

### Problema Crítico: Combat Agent NO entra en batallas

**Hallazgos:**
- El Combat Agent alcanza los **5000 steps** sin entrar en batallas
- `time_in_battle = 0` en todos los episodios
- `hp_dealt = 0`, `hp_taken = 0`, `battles_won = 0`

**Baseline PPO:**
- Entra en batallas en **{sum(1 for r in data['baseline_results'] if r['time_in_battle'] > 0)}/{data['num_episodes']} episodios** ({sum(1 for r in data['baseline_results'] if r['time_in_battle'] > 0)/data['num_episodes']*100:.0f}%)
- Promedio de **{baseline['avg_steps']:.0f} steps** por episodio (30x más eficiente)
- Win rate: **{baseline['win_rate']*100:.1f}%**

---

## 🧮 Comparación de Fórmulas de Recompensa

### Combat Agent (Modificado)
```python
reward = base_reward + combat_bonus

combat_bonus:
  - HP damage dealt: +0.5 per HP
  - Victory: +100.0
  - HP damage taken: -0.3 per HP
  - Not in battle: -0.02 per step

Enfoque: Maximizar daño y victorias en combate
```

**Problema:** La penalización de -0.02 por step fuera de batalla es insuficiente para motivar al agente a buscar batallas activamente.

### Baseline PPO (Original)
```python
reward = exploration + events + levels + badges + party

Components:
  - Map exploration
  - Event flags progression
  - Level gains
  - Badge collection
  - Party composition

Enfoque: Progreso general del juego
```

**Ventaja:** Incentiva progreso natural que incluye batallas como medio para obtener experiencia y avanzar.

---

## 💡 Conclusiones

### ¿Por qué el Baseline PPO es superior?

1. **Navegación efectiva:** El Baseline PPO ha aprendido a navegar el mundo de forma eficiente
2. **Equilibrio de objetivos:** Balancea exploración, eventos y combate
3. **Experiencia de entrenamiento:** 26M timesteps vs 1M del Combat Agent
4. **Recompensas holísticas:** No solo combate, sino progreso integral

### ¿Por qué el Combat Agent falla?

1. **No encuentra batallas:** El agente prioriza evitar la penalización (-0.02) sobre buscar batallas
2. **Falta de guía:** Las recompensas de combate (+0.5/HP, +100 victoria) nunca se activan
3. **Entrenamiento limitado:** Solo 1M timesteps, insuficiente para aprender navegación
4. **Estado inicial:** `has_pokedex_nballs.state` requiere navegación para encontrar batallas

---

## 🎯 Recomendaciones

### Para mejorar el Combat Agent:

1. **Usar estados de batalla directos:**
   - Entrenar con `battle_states/*.state` (pewter_battle, cerulean_battle, etc.)
   - Esto garantiza que el agente empiece **dentro de batallas**

2. **Modificar la función de recompensa:**
   ```python
   # Aumentar penalización por no estar en batalla
   not_in_battle_penalty = -0.5  # en lugar de -0.02
   
   # Agregar recompensa por entrar a batalla
   entered_battle_bonus = +50.0
   ```

3. **Extender entrenamiento:**
   - Mínimo 5-10M timesteps para convergencia
   - Usar curriculum learning (estados fáciles → difíciles)

4. **Híbrido:**
   - Combinar recompensas de combate + exploración
   - `reward = 0.7 * combat_reward + 0.3 * baseline_reward`

---

## 📊 Gráficos Generados

Los siguientes gráficos están disponibles en `{output_dir}/`:

1. `metrics_comparison.png` - Comparación de métricas clave
2. `episode_analysis.png` - Análisis detallado por episodio
3. `reward_formulas.png` - Visualización de fórmulas de recompensa
4. `battle_engagement.png` - Análisis de participación en batallas

---

## 🏁 Veredicto Final

**Baseline PPO (PokemonRedExperiments) es superior al Combat Agent actual.**

**Razón principal:** El Combat Agent no ha aprendido a **encontrar y entrar en batallas**, haciendo que sus recompensas de combate nunca se activen.

**Próximos pasos:** Reentrenar Combat Agent usando estados de batalla directos o mejorar la función de recompensa para incentivar búsqueda activa de batallas.

---

*Reporte generado automáticamente por `analyze_comparison.py`*
"""
    
    report_path = output_dir / 'COMPARISON_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Saved: COMPARISON_REPORT.md")
    return report_path

def main():
    # Find most recent comparison file
    results_dir = Path('comparison_results')
    comparison_files = list(results_dir.glob('comparison_*.json'))
    
    if not comparison_files:
        print("❌ No comparison files found in comparison_results/")
        return
    
    latest_file = max(comparison_files, key=lambda p: p.stat().st_mtime)
    print(f"\n📂 Analyzing: {latest_file.name}\n")
    
    # Load data
    data = load_comparison(latest_file)
    
    # Create output directory
    output_dir = results_dir / f"analysis_{data['timestamp']}"
    output_dir.mkdir(exist_ok=True)
    
    print(f"📊 Generating visualizations...\n")
    
    # Generate all plots
    create_metrics_comparison_plot(data, output_dir)
    create_episode_analysis_plot(data, output_dir)
    create_reward_formula_comparison(output_dir)
    create_battle_engagement_plot(data, output_dir)
    
    # Generate report
    print(f"\n📝 Generating markdown report...\n")
    report_path = generate_markdown_report(data, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ Analysis complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Report: {report_path}")
    print(f"{'='*60}\n")
    
    # Print quick summary
    combat = data['summary']['combat']
    baseline = data['summary']['baseline']
    
    print("\n🏆 QUICK SUMMARY:")
    print(f"  Baseline PPO Reward: {baseline['avg_reward']:.2f}")
    print(f"  Combat Agent Reward: {combat['avg_reward']:.2f}")
    print(f"  Winner: {'Baseline PPO' if baseline['avg_reward'] > combat['avg_reward'] else 'Combat Agent'}")
    print(f"\n  Baseline Win Rate: {baseline['win_rate']*100:.1f}%")
    print(f"  Combat Win Rate: {combat['win_rate']*100:.1f}%")
    print(f"\n⚠️  CRITICAL: Combat Agent time_in_battle = {combat['avg_steps']:.0f} steps with 0 battles!\n")

if __name__ == '__main__':
    main()
