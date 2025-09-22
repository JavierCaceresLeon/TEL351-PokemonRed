"""
Interactive Epsilon Variable Testing for Pokemon Red
===================================================

Script interactivo para probar diferentes valores de epsilon en tiempo real
con el entorno real de Pokemon Red v2.
"""

import sys
import os
import time
import threading
import signal
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
    print(f"Warning: v2 dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

class InteractiveEpsilonTester:
    def __init__(self):
        self.agent = None
        self.env = None
        self.running = False
        self.paused = False
        self.step_count = 0
        self.total_reward = 0
        
    def setup_environment(self):
        """Initialize Pokemon Red environment"""
        print(" Inicializando entorno Pokemon Red...")
        
        env_config = {
            'headless': False,
            'save_final_state': True,
            'early_stop': False,
            'action_freq': 24,
            'init_state': '../init.state',
            'max_steps': 2**20,
            'print_rewards': True,
            'save_video': False,
            'fast_video': True,
            'session_path': './interactive_epsilon_session',
            'gb_path': '../PokemonRed.gb',
            'debug': False,
            'sim_frame_dist': 2_000_000.0
        }
        
        self.env = RedGymEnv(env_config)
        self.env = StreamWrapper(
            self.env, 
            session_path="./interactive_epsilon_session",
            save_video=False
        )
        
        print(" Entorno inicializado correctamente")
        
    def setup_agent(self, initial_epsilon=0.1):
        """Initialize epsilon variable agent"""
        if self.env is None:
            raise ValueError("Environment must be initialized first")
            
        self.agent = VariableEpsilonGreedyAgent(
            self.env, 
            epsilon=initial_epsilon,
            alpha=0.1,
            gamma=0.95
        )
        
        print(f" Agente inicializado con epsilon={initial_epsilon}")
        
    def print_controls(self):
        """Print available controls"""
        print("\n" + "="*60)
        print(" CONTROLES INTERACTIVOS")
        print("="*60)
        print(" EPSILON CONTROLS:")
        print("  1-7: Cambiar a preset (1=pure_exploitation, 7=very_high_exploration)")
        print("  +/-: Aumentar/Disminuir epsilon en 0.1")
        print("  r: Reset epsilon a 0.1 (recomendado)")
        print()
        print("  GAME CONTROLS:")
        print("  SPACE: Pausar/Reanudar")
        print("  s: Mostrar estadísticas")
        print("  c: Limpiar Q-table")
        print("  q: Salir")
        print()
        print(" PRESETS DISPONIBLES:")
        for i, (name, epsilon) in enumerate(EPSILON_CONFIGS.items(), 1):
            print(f"  {i}: {name} (ε={epsilon})")
        print("="*60)
        
    def handle_input(self):
        """Handle user input in separate thread"""
        while self.running:
            try:
                user_input = input().strip().lower()
                
                if user_input == 'q':
                    print(" Saliendo...")
                    self.running = False
                    
                elif user_input == ' ':
                    self.paused = not self.paused
                    print(f"  {'Pausado' if self.paused else 'Reanudado'}")
                    
                elif user_input == 's':
                    self.print_statistics()
                    
                elif user_input == 'c':
                    self.agent.q_table.clear()
                    print("  Q-table limpiada")
                    
                elif user_input == 'r':
                    self.agent.set_epsilon(0.1)
                    print(" Epsilon reset a 0.1 (recomendado)")
                    
                elif user_input == '+':
                    new_epsilon = min(1.0, self.agent.epsilon + 0.1)
                    self.agent.set_epsilon(new_epsilon)
                    
                elif user_input == '-':
                    new_epsilon = max(0.0, self.agent.epsilon - 0.1)
                    self.agent.set_epsilon(new_epsilon)
                    
                elif user_input.isdigit():
                    preset_num = int(user_input)
                    presets = list(EPSILON_CONFIGS.items())
                    if 1 <= preset_num <= len(presets):
                        preset_name, epsilon = presets[preset_num - 1]
                        self.agent.set_epsilon(epsilon)
                        print(f" Cambiado a preset '{preset_name}' (ε={epsilon})")
                    else:
                        print(f" Preset inválido. Usar 1-{len(presets)}")
                        
                elif user_input:
                    try:
                        epsilon = float(user_input)
                        if 0.0 <= epsilon <= 1.0:
                            self.agent.set_epsilon(epsilon)
                            print(f" Epsilon cambiado a {epsilon}")
                        else:
                            print(" Epsilon debe estar entre 0.0 y 1.0")
                    except ValueError:
                        print(f" Comando no reconocido: {user_input}")
                        
            except EOFError:
                break
            except Exception as e:
                print(f" Error en input: {e}")
                
    def print_statistics(self):
        """Print current agent statistics"""
        if self.agent:
            stats = self.agent.get_statistics()
            print(f"\n ESTADÍSTICAS (Step {self.step_count}):")
            print(f"  Epsilon actual: {stats['epsilon']:.3f}")
            print(f"  Exploración: {stats['exploration_rate']:.1%}")
            print(f"  Explotación: {stats['exploitation_rate']:.1%}")
            print(f"  Recompensa promedio: {stats['avg_reward']:.3f}")
            print(f"  Tamaño Q-table: {stats['q_table_size']}")
            print(f"  Recompensa total: {self.total_reward:.2f}")
            print()
            
    def run_interactive_session(self):
        """Run the interactive epsilon testing session"""
        if not DEPENDENCIES_AVAILABLE:
            print(" No se pueden cargar las dependencias v2")
            print(" Ejecuta en su lugar: python test_epsilon_simple.py")
            return
            
        print(" Iniciando sesión interactiva de epsilon variables...")
        
        try:
            # Setup
            self.setup_environment()
            self.setup_agent(initial_epsilon=0.1)  # Start with optimal epsilon
            
            # Print controls
            self.print_controls()
            
            # Start input handler thread
            self.running = True
            input_thread = threading.Thread(target=self.handle_input, daemon=True)
            input_thread.start()
            
            # Initialize environment
            obs = self.env.reset()
            self.total_reward = 0
            self.step_count = 0
            
            print("\n ¡Sesión iniciada! El agente está jugando Pokemon Red...")
            print(" Escribe comandos para cambiar epsilon dinámicamente")
            print(" Epsilon inicial: 0.1 (óptimo según tests)")
            
            start_time = time.time()
            last_stats_time = start_time
            
            # Main game loop
            while self.running:
                if not self.paused:
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
                    
                    # Reset if episode ends
                    if done or truncated:
                        print(f" Episodio terminado en step {self.step_count}")
                        obs = self.env.reset()
                        self.total_reward = 0
                    
                    # Print periodic stats
                    current_time = time.time()
                    if current_time - last_stats_time > 30:  # Every 30 seconds
                        self.print_statistics()
                        last_stats_time = current_time
                
                else:
                    time.sleep(0.1)  # Small delay when paused
                    
        except KeyboardInterrupt:
            print("\n Sesión interrumpida por el usuario")
            
        except Exception as e:
            print(f"\n Error durante la sesión: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.running = False
            if self.env:
                self.env.close()
            print(" Sesión finalizada")

def run_simplified_interactive():
    """Run interactive session with mock environment if v2 not available"""
    print(" Sesión Interactiva Epsilon Variables (Modo Simplificado)")
    print("="*60)
    
    # Mock environment for testing
    class MockEnv:
        def reset(self):
            return [0] * 100
        def step(self, action):
            import random
            reward = random.uniform(0, 2)
            return [0] * 100, reward, False, False, {}
        def close(self):
            pass
    
    env = MockEnv()
    agent = VariableEpsilonGreedyAgent(env, epsilon=0.1)
    
    print(" Comandos disponibles:")
    print("  1-7: Presets de epsilon")
    print("  +/-: Aumentar/Disminuir epsilon")
    print("  s: Estadísticas")
    print("  q: Salir")
    print("\n Agente iniciado con epsilon=0.1")
    
    # Simulate some steps
    obs = env.reset()
    step_count = 0
    
    while True:
        # Simulate steps automatically
        for _ in range(10):
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            agent.update_q_value(obs, action, reward, next_obs)
            obs = next_obs
            step_count += 1
        
        # Get user input
        try:
            user_input = input(f"Step {step_count} (ε={agent.epsilon:.3f}) > ").strip()
            
            if user_input == 'q':
                break
            elif user_input == 's':
                stats = agent.get_statistics()
                print(f"  Epsilon: {stats['epsilon']:.3f}")
                print(f"  Exploración: {stats['exploration_rate']:.1%}")
                print(f"  Q-table: {stats['q_table_size']} estados")
            elif user_input.isdigit():
                presets = list(EPSILON_CONFIGS.values())
                preset_num = int(user_input) - 1
                if 0 <= preset_num < len(presets):
                    agent.set_epsilon(presets[preset_num])
            elif user_input == '+':
                agent.set_epsilon(min(1.0, agent.epsilon + 0.1))
            elif user_input == '-':
                agent.set_epsilon(max(0.0, agent.epsilon - 0.1))
                
        except (KeyboardInterrupt, EOFError):
            break
    
    print(" Sesión simplificada finalizada")

if __name__ == "__main__":
    print(" Interactive Epsilon Variable Tester")
    print("="*60)
    
    if DEPENDENCIES_AVAILABLE:
        print(" Dependencias v2 disponibles - Modo completo")
        tester = InteractiveEpsilonTester()
        tester.run_interactive_session()
    else:
        print("  Dependencias v2 no disponibles - Modo simplificado")
        run_simplified_interactive()