"""
Simple Interactive Epsilon Tester
================================

Versión simplificada para probar epsilon variants de forma interactiva
sin dependencias complejas del entorno v2.
"""

import sys
import os
import time
import random
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from epsilon_greedy.epsilon_variable_agent import VariableEpsilonGreedyAgent, EPSILON_CONFIGS
import numpy as np

class SimpleMockEnv:
    """Entorno mock que simula comportamiento de Pokemon Red"""
    
    def __init__(self):
        self.step_count = 0
        self.current_area = "pallet_town"
        self.areas = ["pallet_town", "route_1", "viridian_city", "route_2", "viridian_forest"]
        self.pokemon_encountered = 0
        
    def reset(self):
        self.step_count = 0
        self.current_area = "pallet_town"
        self.pokemon_encountered = 0
        return self._get_observation()
    
    def _get_observation(self):
        """Genera observación mock que simula pantalla del juego"""
        # Simula variación en la pantalla basada en área y steps
        base_obs = np.random.randint(50, 200, (100,), dtype=np.uint8)
        
        # Añade variación según área
        area_modifier = hash(self.current_area) % 100
        base_obs[0:10] = area_modifier
        
        return base_obs
    
    def step(self, action):
        self.step_count += 1
        
        # Simula diferentes escenarios basados en la acción
        reward = 0.1  # Recompensa base por explorar
        
        # Movimiento (acciones 0-3)
        if action in [0, 1, 2, 3]:
            # Chance de cambiar de área
            if random.random() < 0.02:  # 2% chance
                old_area = self.current_area
                self.current_area = random.choice(self.areas)
                if self.current_area != old_area:
                    reward += 5.0  # Recompensa por descubrir nueva área
                    print(f"🗺️  Área cambiada: {old_area} → {self.current_area}")
        
        # Acción A (acción 4) - interactuar
        elif action == 4:
            if random.random() < 0.05:  # 5% chance
                self.pokemon_encountered += 1
                reward += 10.0  # Gran recompensa por encontrar Pokemon
                print(f"🐾 ¡Pokemon encontrado! Total: {self.pokemon_encountered}")
        
        # Acción B (acción 5) - menú
        elif action == 5:
            reward -= 0.5  # Pequeña penalización por usar menú mucho
        
        # Start/Select (acciones 6-7)
        elif action in [6, 7]:
            reward -= 0.2  # Penalización por acciones no útiles
        
        # Añade variabilidad
        reward += random.uniform(-0.2, 0.2)
        
        # El episodio nunca termina en esta versión simplificada
        done = False
        truncated = False
        
        info = {
            "step": self.step_count,
            "area": self.current_area,
            "pokemon": self.pokemon_encountered
        }
        
        return self._get_observation(), reward, done, truncated, info

class InteractiveEpsilonDemo:
    def __init__(self):
        self.env = SimpleMockEnv()
        self.agent = None
        self.step_count = 0
        self.total_reward = 0
        self.session_start = time.time()
        
    def initialize_agent(self, epsilon=0.1):
        """Inicializa el agente con epsilon específico"""
        self.agent = VariableEpsilonGreedyAgent(self.env, epsilon=epsilon)
        print(f"🤖 Agente inicializado con epsilon={epsilon}")
        
    def print_menu(self):
        """Muestra el menú de opciones"""
        print("\n" + "="*70)
        print("EPSILON INTERACTIVE TESTER")
        print("="*70)
        print("CAMBIAR EPSILON:")
        print("  1: pure_exploitation (ε=0.01)    2: low_exploration (ε=0.1)")
        print("  3: moderate_exploitation (ε=0.3) 4: balanced (ε=0.5)")
        print("  5: high_exploration (ε=0.7)      6: very_high_exploration (ε=0.9)")
        print("  +: Aumentar epsilon (+0.1)       -: Disminuir epsilon (-0.1)")
        print("  c: Epsilon personalizado")
        print()
        print("ACCIONES:")
        print("  r: Ejecutar pasos automáticamente (recomendado)")
        print("  s: Ver estadísticas detalladas")
        print("  m: Mostrar este menú")
        print("  q: Salir")
        print("="*70)
        
    def print_statistics(self):
        """Muestra estadísticas detalladas"""
        if not self.agent:
            return
            
        stats = self.agent.get_statistics()
        elapsed = time.time() - self.session_start
        
        print(f"\nESTADÍSTICAS DETALLADAS (Step {self.step_count})")
        print("-" * 50)
        print(f"Configuración:")
        print(f"   Epsilon actual: {stats['epsilon']:.3f}")
        print(f"   Tiempo transcurrido: {elapsed:.1f}s")
        print()
        print(f"Comportamiento:")
        print(f"   Acciones de exploración: {stats['exploration_rate']:.1%}")
        print(f"   Acciones de explotación: {stats['exploitation_rate']:.1%}")
        print(f"   Total acciones: {stats['total_actions']}")
        print()
        print(f"Rendimiento:")
        print(f"   Recompensa promedio: {stats['avg_reward']:.3f}")
        print(f"   Recompensa total: {self.total_reward:.2f}")
        print(f"   Estados en Q-table: {stats['q_table_size']}")
        print(f"   Recompensas recientes: {stats['recent_rewards']}")
        print("-" * 50)
        
    def run_steps(self, num_steps=50):
        """Ejecuta varios pasos automáticamente"""
        if not self.agent:
            return
            
        print(f"🏃 Ejecutando {num_steps} pasos automáticamente...")
        
        obs = self.env.reset() if self.step_count == 0 else self.env._get_observation()
        step_rewards = []
        
        for i in range(num_steps):
            # Agente selecciona acción
            action = self.agent.select_action(obs)
            
            # Entorno ejecuta paso
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Agente aprende
            self.agent.update_q_value(obs, action, reward, next_obs)
            
            # Actualizar contadores
            self.step_count += 1
            self.total_reward += reward
            step_rewards.append(reward)
            obs = next_obs
            
            # Mostrar progreso cada 10 pasos
            if (i + 1) % 10 == 0:
                avg_reward = np.mean(step_rewards[-10:])
                print(f"   Steps {i-8:2d}-{i+1:2d}: Recompensa promedio = {avg_reward:.3f}")
        
        print(f"{num_steps} pasos completados")
        
        # Mostrar resumen rápido
        stats = self.agent.get_statistics()
        print(f"   Epsilon: {stats['epsilon']:.3f} | "
              f"Exploración: {stats['exploration_rate']:.1%} | "
              f"Q-table: {stats['q_table_size']} estados")
        
    def change_epsilon_preset(self, preset_num):
        """Cambia epsilon usando preset"""
        presets = list(EPSILON_CONFIGS.items())
        if 1 <= preset_num <= len(presets):
            preset_name, epsilon = presets[preset_num - 1]
            self.agent.set_epsilon(epsilon)
            print(f"Epsilon cambiado a '{preset_name}' (ε={epsilon})")
        else:
            print(f"Preset inválido. Usar 1-{len(presets)}")
    
    def adjust_epsilon(self, delta):
        """Ajusta epsilon en incrementos"""
        new_epsilon = max(0.0, min(1.0, self.agent.epsilon + delta))
        self.agent.set_epsilon(new_epsilon)
        print(f"Epsilon ajustado a {new_epsilon:.3f}")
    
    def set_custom_epsilon(self):
        """Permite establecer epsilon personalizado"""
        try:
            epsilon_str = input("Ingresa valor de epsilon (0.0-1.0): ")
            epsilon = float(epsilon_str)
            if 0.0 <= epsilon <= 1.0:
                self.agent.set_epsilon(epsilon)
                print(f"Epsilon personalizado: {epsilon}")
            else:
                print("Epsilon debe estar entre 0.0 y 1.0")
        except ValueError:
            print("Valor inválido")
    
    def run_interactive_session(self):
        """Ejecuta la sesión interactiva principal"""
        print("Iniciando Demo Interactivo de Epsilon Variables")
        print("="*70)
        
        # Inicializar agente
        self.initialize_agent(epsilon=0.1)  # Empezar con epsilon óptimo
        
        # Mostrar menú inicial
        self.print_menu()
        
        print(f"\nConsejo: Empieza escribiendo 'r' para ver el agente en acción")
        print(f"   Epsilon inicial: 0.1 (óptimo según nuestros tests)")
        
        while True:
            try:
                # Mostrar prompt con información actual
                epsilon = self.agent.epsilon if self.agent else 0.0
                user_input = input(f"\n[Step {self.step_count}] (ε={epsilon:.3f}) > ").strip().lower()
                
                if user_input == 'q':
                    print("¡Hasta luego!")
                    break
                
                elif user_input == 'm':
                    self.print_menu()
                
                elif user_input == 's':
                    self.print_statistics()
                
                elif user_input == 'r':
                    self.run_steps(50)
                
                elif user_input == '+':
                    self.adjust_epsilon(0.1)
                
                elif user_input == '-':
                    self.adjust_epsilon(-0.1)
                
                elif user_input == 'c':
                    self.set_custom_epsilon()
                
                elif user_input.isdigit():
                    preset_num = int(user_input)
                    self.change_epsilon_preset(preset_num)
                
                elif user_input.replace('.', '').isdigit():
                    # Direct epsilon input
                    try:
                        epsilon = float(user_input)
                        if 0.0 <= epsilon <= 1.0:
                            self.agent.set_epsilon(epsilon)
                            print(f"Epsilon directo: {epsilon}")
                        else:
                            print("Epsilon debe estar entre 0.0 y 1.0")
                    except ValueError:
                        print("Valor inválido")
                
                elif user_input == '':
                    # Enter solo = ejecutar unos pocos pasos
                    self.run_steps(10)
                
                else:
                    print(f"Comando no reconocido: '{user_input}'")
                    print("Escribe 'm' para ver el menú")
                    
            except (KeyboardInterrupt, EOFError):
                print("\nSesión interrumpida")
                break
        
        print("Demo finalizado")

if __name__ == "__main__":
    demo = InteractiveEpsilonDemo()
    demo.run_interactive_session()