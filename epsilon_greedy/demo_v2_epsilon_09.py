"""
Demo Automático V2 Epsilon 0.9 - Entorno Real Pokemon Red
=========================================================

Demo automático que muestra epsilon = 0.9 en el entorno real de Pokemon Red
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
    print("   Fallback: Use demo_epsilon_09.py for mock environment testing.")
    DEPENDENCIES_AVAILABLE = False

class AutoDemoV2_09:
    """Demo automático V2 con epsilon fijo de 0.9"""
    
    def __init__(self):
        self.agent = None
        self.env = None
        self.step_count = 0
        self.total_reward = 0
        self.session_start = time.time()
        self.EPSILON = 0.9  # Epsilon fijo muy alto
        
    def setup_environment(self):
        """Initialize Pokemon Red V2 environment optimized for high exploration"""
        print("🎮 Inicializando entorno Pokemon Red V2 (Modo Exploración Extrema)...")
        
        # Configuración optimizada para exploración masiva
        env_config = {
            'headless': False,  # Mostrar ventana para ver el caos
            'save_final_state': True,
            'early_stop': False,
            'action_freq': 18,  # Más rápido para mostrar más exploración
            'init_state': '../init.state',
            'max_steps': 2**21,  # Muchos pasos para exploración extrema
            'print_rewards': False,  # Evitar spam
            'save_video': True,  # Grabar la locura
            'fast_video': True,
            'session_path': f'./demo_v2_epsilon_09_chaos_{int(time.time())}',
            'gb_path': '../PokemonRed.gb',
            'debug': False,
            'sim_frame_dist': 3_000_000.0  # Más frames para exploración
        }
        
        self.env = RedGymEnv(env_config)
        self.env = StreamWrapper(
            self.env, 
            session_path=env_config['session_path'],
            save_video=True  # Capturar el comportamiento caótico
        )
        
        print("✅ Entorno V2 inicializado para EXPLORACIÓN EXTREMA")
        print(f"   🎥 Grabando video: {env_config['save_video']}")
        print(f"   ⚡ Velocidad aumentada: {env_config['action_freq']} frames/acción")
        print(f"   📁 Sesión: {env_config['session_path']}")
        
    def setup_agent(self):
        """Initialize epsilon variable agent with very high epsilon"""
        if self.env is None:
            raise ValueError("Environment must be initialized first")
            
        self.agent = VariableEpsilonGreedyAgent(
            self.env, 
            epsilon=self.EPSILON,
            alpha=0.15,  # Aprendizaje más rápido para exploración caótica
            gamma=0.9    # Descuento menor para recompensas inmediatas
        )
        
        print(f"🤖 Agente V2 inicializado con epsilon={self.EPSILON}")
        print("   → 90% exploración caótica, 10% explotación")
        print("   → Comportamiento altamente errático en entorno real")
        print("   → ⚠️  Advertencia: Comportamiento impredecible")
        
    def print_header(self):
        """Print demo information"""
        print("=" * 100)
        print("DEMO AUTOMÁTICO V2: EPSILON 0.9 - EXPLORACIÓN EXTREMA EN POKEMON RED REAL")
        print("=" * 100)
        print("🌪️  Este demo muestra epsilon = 0.9 ejecutándose en el emulador real de Pokemon Red")
        print("• 90% exploración - acciones casi completamente aleatorias")
        print("• 10% explotación - ocasional uso de conocimiento aprendido")
        print("• 🎮 Emulador Game Boy visible para observar comportamiento caótico")
        print("• 🎥 Grabación automática del comportamiento errático")
        print("• ⚠️  ADVERTENCIA: Comportamiento muy impredecible y errático")
        print("=" * 100)
        
    def print_statistics(self, show_detailed=False):
        """Print current agent statistics with chaos indicators"""
        if not self.agent:
            return
            
        stats = self.agent.get_statistics()
        elapsed = time.time() - self.session_start
        
        # Calculate chaos metrics
        chaos_level = "EXTREMO" if stats['exploration_rate'] > 0.85 else "ALTO"
        stability = "MUY BAJA" if stats['exploration_rate'] > 0.85 else "BAJA"
        
        print(f"\n📊 ESTADÍSTICAS CAÓTICAS V2 (Step {self.step_count})")
        print("-" * 80)
        print(f"Configuración:")
        print(f"   🌪️  Epsilon fijo: {stats['epsilon']:.1f} (Exploración Extrema)")
        print(f"   ⏱️  Tiempo transcurrido: {elapsed/60:.1f} minutos")
        print(f"   🎮 Entorno: Pokemon Red V2 (PyBoy) - MODO CAOS")
        print()
        print(f"Comportamiento Caótico:")
        print(f"   🔍 Exploración: {stats['exploration_rate']:.1%} (Nivel: {chaos_level})")
        print(f"   🎯 Explotación: {stats['exploitation_rate']:.1%}")
        print(f"   📊 Total acciones: {stats['total_actions']}")
        print(f"   🌪️  Nivel de caos: {chaos_level}")
        print(f"   📉 Estabilidad: {stability}")
        print()
        print(f"Aprendizaje Caótico:")
        print(f"   💰 Recompensa promedio: {stats['avg_reward']:.3f}")
        print(f"   🏆 Recompensa total: {self.total_reward:.2f}")
        print(f"   🧠 Estados Q-table: {stats['q_table_size']} (Muy diversa)")
        print(f"   ⚡ Pasos/segundo: {self.step_count/elapsed:.2f}")
        
        if show_detailed:
            print(f"   📋 Recompensas recientes: {stats['recent_rewards']}")
            print(f"   🎲 Predicibilidad: MÍNIMA (Epsilon 0.9)")
        print("-" * 80)
        
    def run_chaos_demo(self, target_minutes=4, stats_interval=45):
        """Execute chaotic exploration demo"""
        print(f"🌪️  Iniciando demo caótico por {target_minutes} minutos")
        print(f"   📊 Estadísticas cada {stats_interval} segundos")
        print(f"   🎥 Grabando comportamiento errático")
        print(f"   ⏹️  Presiona Ctrl+C para detener el caos")
        print()
        
        # Initialize environment
        obs = self.env.reset()
        start_time = time.time()
        end_time = start_time + (target_minutes * 60)
        last_stats_time = start_time
        
        # Chaos tracking variables
        action_chaos_sequence = []
        direction_changes = 0
        last_action = None
        chaos_events = []
        
        print("🎯 CAOS INICIADO - ¡Observa la locura en el emulador Game Boy!")
        print("   🌪️  El agente está explorando caóticamente...")
        print("   📈 Cada acción tiene 90% probabilidad de ser aleatoria")
        
        try:
            while time.time() < end_time:
                # Agent selects action (90% exploration, 10% exploitation)
                action = self.agent.select_action(obs)
                
                # Detect direction changes (chaos indicator)
                if last_action is not None and action != last_action and action < 4 and last_action < 4:
                    direction_changes += 1
                    if direction_changes % 20 == 0:  # Every 20 direction changes
                        chaos_events.append(f"Direction chaos burst at step {self.step_count}")
                
                action_chaos_sequence.append(action)
                last_action = action
                
                # Environment step
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                # Agent learning (even in chaos, it tries to learn)
                self.agent.update_q_value(obs, action, reward, next_obs)
                
                # Update tracking
                self.total_reward += reward
                self.step_count += 1
                obs = next_obs
                
                # Detect special chaos events
                if reward > 5.0:
                    chaos_events.append(f"Chaos reward spike: {reward:.2f} at step {self.step_count}")
                
                # Handle episode completion
                if done or truncated:
                    chaos_events.append(f"Chaos episode end at step {self.step_count}")
                    print(f"🔄 Episodio caótico completado, reiniciando el caos...")
                    obs = self.env.reset()
                
                # Print periodic chaos analysis
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    elapsed = current_time - start_time
                    remaining = (end_time - current_time) / 60
                    
                    print(f"\n⏰ Progreso Caótico: {elapsed/60:.1f}/{target_minutes} min | "
                          f"Restante: {remaining:.1f} min")
                    
                    # Chaos action analysis
                    if action_chaos_sequence:
                        action_names = ['↑', '↓', '←', '→', 'A', 'B', 'START', 'SELECT']
                        recent_actions = action_chaos_sequence[-100:]  # Last 100 actions
                        
                        print("🎲 ANÁLISIS DE CAOS - Últimas 100 acciones:")
                        action_counts = [recent_actions.count(i) for i in range(8)]
                        for i, (name, count) in enumerate(zip(action_names, action_counts)):
                            if count > 0:
                                percentage = count/len(recent_actions)*100
                                chaos_bar = "🔥" * int(percentage/10)
                                print(f"     {name:6s}: {count:2d} ({percentage:4.1f}%) {chaos_bar}")
                        
                        # Chaos metrics
                        print(f"\n🌪️  MÉTRICAS DE CAOS:")
                        print(f"     Cambios direccionales: {direction_changes}")
                        print(f"     Eventos caóticos: {len(chaos_events)}")
                        uniformity = max(action_counts) - min(action_counts)
                        print(f"     Uniformidad caótica: {uniformity} (menor = más caótico)")
                    
                    self.print_statistics()
                    last_stats_time = current_time
                
                # Minimal delay (chaos is fast!)
                time.sleep(0.0005)
                
        except KeyboardInterrupt:
            print("\n⏹️  ¡CAOS DETENIDO! Usuario interrumpió la locura")
        
        final_elapsed = time.time() - start_time
        print(f"\n🏁 CAOS FINALIZADO después de {final_elapsed/60:.1f} minutos")
        print(f"   🌪️  Total de eventos caóticos: {len(chaos_events)}")
        print(f"   📊 Cambios direccionales: {direction_changes}")
        
        # Chaos event summary
        if chaos_events:
            print(f"\n🎪 EVENTOS CAÓTICOS DESTACADOS:")
            for event in chaos_events[-5:]:  # Last 5 events
                print(f"     • {event}")
        
        return {
            'duration': final_elapsed,
            'steps': self.step_count,
            'chaos_events': len(chaos_events),
            'direction_changes': direction_changes,
            'total_reward': self.total_reward
        }
        
    def run_discovery_frenzy(self):
        """Execute discovery-focused demo with extreme exploration"""
        print("🔍 FRENESÍ DE DESCUBRIMIENTO - Epsilon 0.9 explorando todo")
        print("=" * 80)
        print("Objetivos del frenesí:")
        print("• Explorar cada rincón posible de forma caótica")
        print("• Interactuar con todo lo que encuentre")
        print("• Descubrir mecánicas ocultas por pura exploración")
        print("• Generar la Q-table más diversa posible")
        print("=" * 80)
        
        obs = self.env.reset()
        start_time = time.time()
        
        # Discovery tracking
        interaction_attempts = 0
        unique_states_found = set()
        discovery_milestones = []
        
        print("🚀 ¡FRENESÍ INICIADO! Exploración caótica extrema...")
        print("   🎮 El agente intentará absolutamente TODO")
        
        target_steps = 1500  # Extended for maximum discovery
        discovery_rewards = []
        
        try:
            for step in range(target_steps):
                # Agent selects chaotic action
                action = self.agent.select_action(obs)
                
                # Count interaction attempts (A and B buttons)
                if action in [4, 5]:  # A or B buttons
                    interaction_attempts += 1
                
                # Environment step  
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                # Track unique states (simplified hash)
                state_hash = hash(str(obs))
                unique_states_found.add(state_hash)
                
                # Agent learning
                self.agent.update_q_value(obs, action, reward, next_obs)
                
                # Update tracking
                self.total_reward += reward
                self.step_count += 1
                obs = next_obs
                
                # Discovery milestone detection
                if reward > 8.0:
                    discovery_milestones.append({
                        'step': step,
                        'reward': reward,
                        'action': action,
                        'unique_states': len(unique_states_found)
                    })
                    print(f"💎 ¡DESCUBRIMIENTO! Step {step}: R={reward:.1f}, "
                          f"Estados únicos: {len(unique_states_found)}")
                
                # Progress tracking every 200 steps  
                if (step + 1) % 200 == 0:
                    action_names = ['↑', '↓', '←', '→', 'A', 'B', 'START', 'SELECT']
                    action_name = action_names[action] if action < len(action_names) else f'A{action}'
                    chaos_indicator = "🌪️" if random.random() < 0.9 else "🎯"
                    
                    discovery_rate = len(discovery_milestones) / (step + 1) * 1000
                    print(f"Step {step+1:4d}: {chaos_indicator} {action_name} → "
                          f"Estados: {len(unique_states_found)} | "
                          f"Descubrimientos: {len(discovery_milestones)} "
                          f"({discovery_rate:.1f}/1000 pasos)")
                
                # Handle episode completion
                if done or truncated:
                    print(f"🔄 Episodio de frenesí completado, continuando exploración...")
                    obs = self.env.reset()
                
        except KeyboardInterrupt:
            print("\n⏹️  Frenesí de descubrimiento interrumpido")
        
        elapsed = time.time() - start_time
        print(f"\n🏆 RESUMEN DEL FRENESÍ:")
        print(f"   ⏱️  Tiempo total: {elapsed/60:.1f} minutos")
        print(f"   🔍 Estados únicos encontrados: {len(unique_states_found)}")
        print(f"   🎯 Intentos de interacción: {interaction_attempts}")
        print(f"   💎 Descubrimientos importantes: {len(discovery_milestones)}")
        print(f"   🧠 Q-table diversidad: {self.agent.get_statistics()['q_table_size']} estados")
        
        # Show best discoveries
        if discovery_milestones:
            best_discoveries = sorted(discovery_milestones, key=lambda x: x['reward'], reverse=True)[:3]
            print(f"\n🌟 MEJORES DESCUBRIMIENTOS:")
            for i, discovery in enumerate(best_discoveries, 1):
                action_names = ['↑', '↓', '←', '→', 'A', 'B', 'START', 'SELECT']
                action_name = action_names[discovery['action']]
                print(f"     {i}. Step {discovery['step']}: {action_name} → "
                      f"Recompensa {discovery['reward']:.1f}")
        
        return len(unique_states_found), len(discovery_milestones), interaction_attempts

def run_fallback_demo():
    """Run fallback demo if V2 dependencies are not available"""
    print("🔄 MODO FALLBACK - V2 no disponible")
    print("=" * 60)
    print("Las dependencias V2 no están disponibles.")
    print("Usa en su lugar:")
    print("  python demo_epsilon_09.py")
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
    
    print("Iniciando Demo Automático V2 - Epsilon 0.9 (EXPLORACIÓN EXTREMA)")
    
    # Create and configure demo
    demo = AutoDemoV2_09()
    demo.print_header()
    
    try:
        # Setup environment and agent
        demo.setup_environment()
        demo.setup_agent()
        
        print("\n🤖 MODO AUTOMÁTICO CAÓTICO V2: Ejecutando demo de 4 minutos...")
        print("   No se requiere entrada del usuario")
        print("   ⚠️  Comportamiento altamente errático esperado")
        print("   Presiona Ctrl+C para detener el caos")
        print()
        
        try:
            # Execute chaotic automatic demo (4 minutes)
            result = demo.run_chaos_demo(target_minutes=4, stats_interval=45)
                
        except (KeyboardInterrupt, EOFError):
            print("\n🤖 Caos detenido por el usuario")
        
        # Final statistics
        print("\n" + "="*100)
        print("RESUMEN FINAL DEL DEMO CAÓTICO V2")
        print("="*100)
        demo.print_statistics(show_detailed=True)
        
        stats = demo.agent.get_statistics()
        print(f"\n💡 CONCLUSIONES EPSILON 0.9 EN ENTORNO REAL:")
        print(f"   🌪️  Exploración observada: {stats['exploration_rate']:.1%}")
        print(f"   🎲 Comportamiento altamente errático y caótico")
        print(f"   🧠 Q-table súper diversa: {stats['q_table_size']} estados únicos")
        print(f"   🔍 Máxima capacidad de descubrimiento")
        print(f"   ⚡ Velocidad: {demo.step_count/(time.time()-demo.session_start):.1f} pasos/segundo")
        print(f"   ⚠️  Requiere reducción gradual de epsilon para estabilizar")
        print(f"   🎥 Video grabado para análisis posterior")
        print("="*100)
        
    except Exception as e:
        print(f"\n❌ Error durante el demo caótico: {e}")
        print("El caos a veces es demasiado para el sistema...")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if demo.env:
            demo.env.close()
        print("🔧 Caos contenido y limpieza completada")

if __name__ == "__main__":
    main()