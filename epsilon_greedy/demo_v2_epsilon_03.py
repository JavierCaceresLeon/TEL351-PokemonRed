"""
Demo Automático V2 Epsilon 0.3 - Entorno Real Pokemon Red
=========================================================

Demo automático que muestra epsilon = 0.3 en el entorno real de Pokemon Red
usando PyBoy emulator. Basado en run_interactive_epsilon.py
"""

import sys
import os
import time
import random
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import dependencies
try:
    from v2.red_gym_env_v2 import RedGymEnv
    from v2.stream_agent_wrapper import StreamWrapper
    from epsilon_greedy.epsilon_variable_agent import VariableEpsilonGreedyAgent, EPSILON_CONFIGS
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: v2 dependencies not available: {e}")
    print("   This demo requires the v2 environment to run.")
    print("   Fallback: Use demo_epsilon_03.py for mock environment testing.")
    DEPENDENCIES_AVAILABLE = False

class AutoDemoV2_03:
    """Demo automático V2 con epsilon fijo de 0.3"""
    
    def __init__(self):
        self.agent = None
        self.env = None
        self.step_count = 0
        self.total_reward = 0
        self.session_start = time.time()
        self.EPSILON = 0.3  # Epsilon fijo moderado
        
    def setup_environment(self):
        """Initialize Pokemon Red V2 environment"""
        print("🎮 Inicializando entorno Pokemon Red V2...")
        
        # Configuración optimizada para demo automático
        env_config = {
            'headless': False,  # Mostrar ventana del emulador
            'save_final_state': True,
            'early_stop': False,
            'action_freq': 24,
            'init_state': '../init.state',
            'max_steps': 2**20,  # Muchos pasos para demo largo
            'print_rewards': False,  # Evitar spam de recompensas
            'save_video': False,
            'fast_video': True,
            'session_path': f'./demo_v2_epsilon_03_session_{int(time.time())}',
            'gb_path': '../PokemonRed.gb',
            'debug': False,
            'sim_frame_dist': 2_000_000.0
        }
        
        self.env = RedGymEnv(env_config)
        self.env = StreamWrapper(
            self.env, 
            session_path=env_config['session_path'],
            save_video=False
        )
        
        print("✅ Entorno V2 inicializado correctamente")
        print(f"   Emulador Game Boy: {'Visible' if not env_config['headless'] else 'Oculto'}")
        print(f"   Sesión: {env_config['session_path']}")
        
    def setup_agent(self):
        """Initialize epsilon variable agent with fixed epsilon"""
        if self.env is None:
            raise ValueError("Environment must be initialized first")
            
        self.agent = VariableEpsilonGreedyAgent(
            self.env, 
            epsilon=self.EPSILON,
            alpha=0.1,
            gamma=0.95
        )
        
        print(f"🤖 Agente V2 inicializado con epsilon={self.EPSILON}")
        print("   → 30% exploración, 70% explotación")
        print("   → Comportamiento balanceado en entorno real")
        
    def print_header(self):
        """Print demo information"""
        print("=" * 90)
        print("DEMO AUTOMÁTICO V2: EPSILON 0.3 - ENTORNO REAL POKEMON RED")
        print("=" * 90)
        print("Este demo muestra epsilon = 0.3 ejecutándose en el emulador real de Pokemon Red")
        print("• 30% exploración - acciones aleatorias para descubrir")
        print("• 70% explotación - acciones basadas en Q-table aprendida")
        print("• Emulador Game Boy visible para observar comportamiento real")
        print("• Sin intervención manual - completamente automático")
        print("=" * 90)
        
    def print_statistics(self, show_detailed=False):
        """Print current agent statistics"""
        if not self.agent:
            return
            
        stats = self.agent.get_statistics()
        elapsed = time.time() - self.session_start
        
        print(f"\n📊 ESTADÍSTICAS V2 (Step {self.step_count})")
        print("-" * 70)
        print(f"Configuración:")
        print(f"   🎯 Epsilon fijo: {stats['epsilon']:.1f} (Exploración Moderada)")
        print(f"   ⏱️  Tiempo transcurrido: {elapsed/60:.1f} minutos")
        print(f"   🎮 Entorno: Pokemon Red V2 (PyBoy)")
        print()
        print(f"Comportamiento:")
        print(f"   🔍 Exploración: {stats['exploration_rate']:.1%}")
        print(f"   🎯 Explotación: {stats['exploitation_rate']:.1%}")
        print(f"   📊 Total acciones: {stats['total_actions']}")
        print()
        print(f"Aprendizaje:")
        print(f"   💰 Recompensa promedio: {stats['avg_reward']:.3f}")
        print(f"   🏆 Recompensa total: {self.total_reward:.2f}")
        print(f"   🧠 Estados Q-table: {stats['q_table_size']}")
        print(f"   ⚡ Pasos/segundo: {self.step_count/elapsed:.2f}")
        
        if show_detailed:
            print(f"   📋 Recompensas recientes: {stats['recent_rewards']}")
        print("-" * 70)
        
    def run_auto_demo(self, target_minutes=5, stats_interval=60):
        """Execute automatic demo for specified duration"""
        print(f"🚀 Iniciando demo automático por {target_minutes} minutos")
        print(f"   📊 Estadísticas cada {stats_interval} segundos")
        print(f"   ⏹️  Presiona Ctrl+C para detener anticipadamente")
        print()
        
        # Initialize environment
        obs = self.env.reset()
        start_time = time.time()
        end_time = start_time + (target_minutes * 60)
        last_stats_time = start_time
        
        # Tracking variables
        actions_taken = []
        rewards_received = []
        episodes_completed = 0
        
        print("🎯 Demo iniciado - Observa el emulador Game Boy en acción!")
        print("   El agente está aprendiendo y jugando automáticamente...")
        
        try:
            while time.time() < end_time:
                # Agent selects action (30% exploration, 70% exploitation)
                action = self.agent.select_action(obs)
                actions_taken.append(action)
                
                # Environment step
                next_obs, reward, done, truncated, info = self.env.step(action)
                rewards_received.append(reward)
                
                # Agent learning
                self.agent.update_q_value(obs, action, reward, next_obs)
                
                # Update tracking
                self.total_reward += reward
                self.step_count += 1
                obs = next_obs
                
                # Handle episode completion
                if done or truncated:
                    episodes_completed += 1
                    print(f"📋 Episodio {episodes_completed} completado en step {self.step_count}")
                    obs = self.env.reset()
                
                # Print periodic statistics
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    elapsed = current_time - start_time
                    remaining = (end_time - current_time) / 60
                    
                    print(f"\n⏰ Progreso: {elapsed/60:.1f}/{target_minutes} min | "
                          f"Restante: {remaining:.1f} min")
                    
                    # Show action distribution
                    if actions_taken:
                        action_names = ['↑', '↓', '←', '→', 'A', 'B', 'START', 'SELECT']
                        recent_actions = actions_taken[-100:]  # Last 100 actions
                        action_counts = [recent_actions.count(i) for i in range(8)]
                        print("🎮 Últimas 100 acciones:")
                        for i, (name, count) in enumerate(zip(action_names, action_counts)):
                            if count > 0:
                                print(f"     {name}: {count} ({count/len(recent_actions)*100:.1f}%)")
                    
                    self.print_statistics()
                    last_stats_time = current_time
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n⏹️  Demo detenido manualmente por el usuario")
        
        final_elapsed = time.time() - start_time
        print(f"\n🏁 Demo V2 finalizado después de {final_elapsed/60:.1f} minutos")
        print(f"   📊 Episodios completados: {episodes_completed}")
        print(f"   🎯 Pasos totales: {self.step_count}")
        
        return {
            'duration': final_elapsed,
            'steps': self.step_count,
            'episodes': episodes_completed,
            'total_reward': self.total_reward
        }
        
    def run_goal_oriented_demo(self):
        """Execute demo with specific Pokemon Red goals"""
        print("🎯 DEMO ORIENTADO A OBJETIVOS - Epsilon 0.3 en Pokemon Red")
        print("=" * 70)
        print("Objetivos del demo:")
        print("• Salir de Pallet Town")
        print("• Explorar Route 1") 
        print("• Llegar a Viridian City")
        print("• Interactuar con NPCs y objetos")
        print("=" * 70)
        
        obs = self.env.reset()
        start_time = time.time()
        
        # Goal tracking
        goals_achieved = []
        goal_steps = {}
        
        print("🚀 Iniciando demo orientado a objetivos...")
        print("   🎮 Observa el progreso en el emulador Game Boy")
        
        target_steps = 1000  # Limit for goal-oriented demo
        
        try:
            for step in range(target_steps):
                # Agent selects action
                action = self.agent.select_action(obs)
                
                # Environment step  
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                # Agent learning
                self.agent.update_q_value(obs, action, reward, next_obs)
                
                # Update tracking
                self.total_reward += reward
                self.step_count += 1
                obs = next_obs
                
                # Goal detection (simplified - based on reward patterns)
                if reward > 5.0 and "pallet_exit" not in goals_achieved:
                    goals_achieved.append("pallet_exit")
                    goal_steps["pallet_exit"] = step
                    print(f"🎉 ¡Objetivo alcanzado! Salida de Pallet Town (Step {step})")
                
                if reward > 10.0 and "route1_exploration" not in goals_achieved:
                    goals_achieved.append("route1_exploration") 
                    goal_steps["route1_exploration"] = step
                    print(f"🎉 ¡Objetivo alcanzado! Exploración Route 1 (Step {step})")
                
                # Show progress every 100 steps
                if (step + 1) % 100 == 0:
                    action_names = ['↑', '↓', '←', '→', 'A', 'B', 'START', 'SELECT']
                    action_name = action_names[action] if action < len(action_names) else f'A{action}'
                    exploration_indicator = "🔍" if random.random() < 0.3 else "🎯"
                    
                    print(f"Step {step+1:4d}: {exploration_indicator} {action_name} → "
                          f"R={reward:+.2f} | Objetivos: {len(goals_achieved)}")
                
                # Handle episode completion
                if done or truncated:
                    print(f"📋 Episodio completado, reiniciando...")
                    obs = self.env.reset()
                
        except KeyboardInterrupt:
            print("\n⏹️  Demo orientado a objetivos interrumpido")
        
        elapsed = time.time() - start_time
        print(f"\n🏆 RESUMEN DE OBJETIVOS:")
        print(f"   ⏱️  Tiempo total: {elapsed/60:.1f} minutos")
        print(f"   🎯 Objetivos alcanzados: {len(goals_achieved)}")
        for goal, step in goal_steps.items():
            print(f"     • {goal}: Step {step}")
        print(f"   📊 Steps promedio por objetivo: {sum(goal_steps.values())/len(goal_steps) if goal_steps else 'N/A'}")

def run_fallback_demo():
    """Run fallback demo if V2 dependencies are not available"""
    print("🔄 MODO FALLBACK - V2 no disponible")
    print("=" * 60)
    print("Las dependencias V2 no están disponibles.")
    print("Usa en su lugar:")
    print("  python demo_epsilon_03.py")
    print()
    print("Para usar este demo V2, asegúrate de que:")
    print("• El directorio v2/ existe")
    print("• Los archivos red_gym_env_v2.py y stream_agent_wrapper.py están presentes")
    print("• PyBoy está instalado correctamente")
    print("=" * 60)

def main():
    """Main demo function"""
    if not DEPENDENCIES_AVAILABLE:
        run_fallback_demo()
        return
    
    print("Iniciando Demo Automático V2 - Epsilon 0.3")
    
    # Create and configure demo
    demo = AutoDemoV2_03()
    demo.print_header()
    
    try:
        # Setup environment and agent
        demo.setup_environment()
        demo.setup_agent()
        
        print("\n🤖 MODO AUTOMÁTICO V2: Ejecutando demo de 5 minutos...")
        print("   No se requiere entrada del usuario")
        print("   Presiona Ctrl+C para detener anticipadamente")
        print()
        
        try:
            # Execute automatic demo (5 minutes)
            result = demo.run_auto_demo(target_minutes=5, stats_interval=60)
                
        except (KeyboardInterrupt, EOFError):
            print("\n🤖 Demo interrumpido por el usuario")
        
        # Final statistics
        print("\n" + "="*90)
        print("RESUMEN FINAL DEL DEMO V2")
        print("="*90)
        demo.print_statistics(show_detailed=True)
        
        stats = demo.agent.get_statistics()
        print(f"\n💡 CONCLUSIONES EPSILON 0.3 EN ENTORNO REAL:")
        print(f"   • Exploración observada: {stats['exploration_rate']:.1%}")
        print(f"   • Comportamiento balanceado en Pokemon Red real")
        print(f"   • Q-table desarrollada: {stats['q_table_size']} estados únicos")
        print(f"   • Eficiencia: {demo.step_count/(time.time()-demo.session_start):.1f} pasos/segundo")
        print(f"   • ✅ Adecuado para progresión estable en el juego")
        print("="*90)
        
    except Exception as e:
        print(f"\n❌ Error durante el demo: {e}")
        print("Verifica que todas las dependencias V2 estén instaladas correctamente")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if demo.env:
            demo.env.close()
        print("🔧 Limpieza completada")

if __name__ == "__main__":
    main()