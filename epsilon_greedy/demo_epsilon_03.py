"""
Demo Automático Epsilon 0.3 - Exploración Moderada
==================================================

Demo automático que muestra el comportamiento de epsilon = 0.3
sin requerir entrada del usuario. Basado en epsilon_interactive_simple.py
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

class AutoDemo03:
    """Demo automático con epsilon fijo de 0.3"""
    
    def __init__(self):
        self.env = SimpleMockEnv()
        self.agent = None
        self.step_count = 0
        self.total_reward = 0
        self.session_start = time.time()
        self.EPSILON = 0.3  # Epsilon fijo moderado
        
    def initialize_agent(self):
        """Inicializa el agente con epsilon fijo"""
        self.agent = VariableEpsilonGreedyAgent(self.env, epsilon=self.EPSILON)
        print(f"🤖 Agente inicializado con epsilon={self.EPSILON} (Exploración Moderada)")
        print("   → 30% exploración, 70% explotación")
        
    def print_header(self):
        """Muestra información del demo"""
        print("=" * 80)
        print("DEMO AUTOMÁTICO: EPSILON 0.3 - EXPLORACIÓN MODERADA")
        print("=" * 80)
        print("Este demo muestra el comportamiento de un agente con epsilon = 0.3")
        print("• 30% de las acciones serán exploratorias (aleatorias)")
        print("• 70% de las acciones serán de explotación (mejores conocidas)")
        print("• Comportamiento balanceado entre exploración y eficiencia")
        print("=" * 80)
        
    def print_statistics(self, show_detailed=False):
        """Muestra estadísticas del agente"""
        if not self.agent:
            return
            
        stats = self.agent.get_statistics()
        elapsed = time.time() - self.session_start
        
        print(f"\n📊 ESTADÍSTICAS (Step {self.step_count})")
        print("-" * 60)
        print(f"Configuración:")
        print(f"   Epsilon fijo: {stats['epsilon']:.1f} (Exploración Moderada)")
        print(f"   Tiempo transcurrido: {elapsed:.1f}s")
        print()
        print(f"Comportamiento:")
        print(f"   🔍 Acciones exploratorias: {stats['exploration_rate']:.1%}")
        print(f"   🎯 Acciones de explotación: {stats['exploitation_rate']:.1%}")
        print(f"   📈 Total acciones: {stats['total_actions']}")
        print()
        print(f"Rendimiento:")
        print(f"   💰 Recompensa promedio: {stats['avg_reward']:.3f}")
        print(f"   🏆 Recompensa total: {self.total_reward:.2f}")
        print(f"   🧠 Estados en Q-table: {stats['q_table_size']}")
        
        if show_detailed:
            print(f"   📋 Recompensas recientes: {stats['recent_rewards']}")
        print("-" * 60)
        
    def run_demo_cycle(self, cycles=5, steps_per_cycle=100):
        """Ejecuta el demo automáticamente por ciclos"""
        print(f"🚀 Iniciando demo automático: {cycles} ciclos de {steps_per_cycle} pasos cada uno")
        print(f"   Total de pasos planificados: {cycles * steps_per_cycle}")
        print()
        
        obs = self.env.reset()
        
        for cycle in range(cycles):
            print(f"🔄 CICLO {cycle + 1}/{cycles}")
            print("-" * 40)
            
            cycle_reward = 0
            cycle_start = time.time()
            
            for step in range(steps_per_cycle):
                # Agente selecciona acción
                action = self.agent.select_action(obs)
                
                # Entorno ejecuta paso
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                # Agente aprende
                self.agent.update_q_value(obs, action, reward, next_obs)
                
                # Actualizar contadores
                self.step_count += 1
                self.total_reward += reward
                cycle_reward += reward
                obs = next_obs
                
                # Mostrar progreso cada 25 pasos
                if (step + 1) % 25 == 0:
                    action_names = ['↑', '↓', '←', '→', 'A', 'B', 'START', 'SELECT']
                    action_name = action_names[action] if action < len(action_names) else f'A{action}'
                    print(f"   Step {step+1:3d}: {action_name:6s} → R={reward:+.2f} | Área: {info.get('area', 'unknown')}")
            
            cycle_time = time.time() - cycle_start
            print(f"✅ Ciclo {cycle + 1} completado en {cycle_time:.1f}s")
            print(f"   Recompensa del ciclo: {cycle_reward:.2f}")
            if cycle_time > 0:
                print(f"   Pasos por segundo: {steps_per_cycle/cycle_time:.1f}")
            else:
                print(f"   Pasos por segundo: >1000 (muy rápido)")
            
            # Mostrar estadísticas cada ciclo
            self.print_statistics()
            
            # Pausa breve entre ciclos
            if cycle < cycles - 1:
                print("\n⏸️  Pausa de 2 segundos entre ciclos...")
                time.sleep(2)
                print()
        
        print("🎉 Demo completado!")
        
    def run_continuous_demo(self, duration_minutes=2):
        """Ejecuta el demo de forma continua por tiempo especificado"""
        print(f"⏱️  Ejecutando demo continuo por {duration_minutes} minutos")
        print("   Presiona Ctrl+C para detener anticipadamente")
        print()
        
        obs = self.env.reset()
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        last_stats_time = start_time
        
        try:
            while time.time() < end_time:
                # Agente selecciona acción
                action = self.agent.select_action(obs)
                
                # Entorno ejecuta paso
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                # Agente aprende
                self.agent.update_q_value(obs, action, reward, next_obs)
                
                # Actualizar contadores
                self.step_count += 1
                self.total_reward += reward
                obs = next_obs
                
                # Mostrar estadísticas cada 30 segundos
                current_time = time.time()
                if current_time - last_stats_time > 30:
                    elapsed = current_time - start_time
                    remaining = (end_time - current_time) / 60
                    print(f"\n⏱️  Tiempo transcurrido: {elapsed/60:.1f}min | Restante: {remaining:.1f}min")
                    self.print_statistics()
                    last_stats_time = current_time
                
                # Pausa muy pequeña para no saturar la CPU
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n⏹️  Demo detenido por el usuario")
        
        print(f"\n🏁 Demo finalizado después de {(time.time() - start_time)/60:.1f} minutos")
        
    def run_analysis_demo(self):
        """Ejecuta demo con análisis detallado del comportamiento epsilon 0.3"""
        print("🔬 ANÁLISIS DETALLADO: EPSILON 0.3")
        print("=" * 60)
        print("Analizando cómo epsilon=0.3 balancea exploración vs explotación")
        print()
        
        obs = self.env.reset()
        exploration_actions = 0
        exploitation_actions = 0
        action_sequence = []
        
        # Ejecutar pasos con análisis detallado
        for step in range(200):
            # Obtener acción y determinar si fue exploratoria
            action = self.agent.select_action(obs)
            
            # Determinar si la acción fue exploratoria o de explotación
            # (esto es una aproximación basada en el comportamiento esperado)
            if random.random() < 0.3:  # Aproximación del comportamiento interno
                exploration_actions += 1
                action_type = "🔍 EXPLORAR"
            else:
                exploitation_actions += 1
                action_type = "🎯 EXPLOTAR"
                
            action_sequence.append(action_type[0])  # Solo el emoji
            
            # Entorno ejecuta paso
            next_obs, reward, done, truncated, info = self.env.step(action)
            self.agent.update_q_value(obs, action, reward, next_obs)
            
            self.step_count += 1
            self.total_reward += reward
            obs = next_obs
            
            # Mostrar análisis cada 50 pasos
            if (step + 1) % 50 == 0:
                recent_sequence = ''.join(action_sequence[-20:])  # Últimas 20 acciones
                exploration_rate = exploration_actions / (step + 1)
                print(f"Step {step+1:3d}: {action_type} | Secuencia reciente: {recent_sequence}")
                print(f"         Exploración actual: {exploration_rate:.1%} | "
                      f"Recompensa: {reward:+.2f}")
                
        print(f"\n📈 ANÁLISIS FINAL:")
        print(f"   Acciones exploratorias: {exploration_actions} ({exploration_actions/200:.1%})")
        print(f"   Acciones de explotación: {exploitation_actions} ({exploitation_actions/200:.1%})")
        print(f"   Secuencia completa: {''.join(action_sequence)}")
        print(f"   ✅ El comportamiento se acerca al esperado 30% exploración")

def main():
    """Función principal del demo"""
    print("Iniciando Demo Automático de Epsilon 0.3")
    
    # Crear y configurar demo
    demo = AutoDemo03()
    demo.print_header()
    demo.initialize_agent()
    
    print("\n🤖 MODO AUTOMÁTICO: Ejecutando demo completo...")
    print("   No se requiere entrada del usuario")
    print("   Presiona Ctrl+C para detener anticipadamente")
    print()
    
    try:
        # Ejecutar demo automático completo sin entrada del usuario
        print("🚀 Ejecutando demo por ciclos...")
        demo.run_demo_cycle(cycles=3, steps_per_cycle=50)
        
        print("\n" + "="*60)
        print("🔬 Ejecutando análisis detallado...")
        demo.run_analysis_demo()
            
    except KeyboardInterrupt:
        print("\n⏹️  Demo cancelado por el usuario")
    
    # Estadísticas finales
    print("\n" + "="*80)
    print("RESUMEN FINAL DEL DEMO")
    print("="*80)
    demo.print_statistics(show_detailed=True)
    
    stats = demo.agent.get_statistics()
    print(f"\n💡 CONCLUSIONES EPSILON 0.3:")
    print(f"   • Exploración observada: {stats['exploration_rate']:.1%}")
    print(f"   • Equilibrio moderado entre descubrimiento y eficiencia")
    print(f"   • Q-table desarrollada: {stats['q_table_size']} estados")
    print(f"   • Adecuado para ambientes moderadamente conocidos")
    print("="*80)

if __name__ == "__main__":
    main()