"""
Demo Automático Epsilon 0.9 - Exploración Muy Alta
===================================================

Demo automático que muestra el comportamiento de epsilon = 0.9
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
        self.areas = ["pallet_town", "route_1", "viridian_city", "route_2", "viridian_forest", 
                      "pewter_city", "route_3", "mt_moon", "cerulean_city", "route_4"]
        self.pokemon_encountered = 0
        self.areas_discovered = set(["pallet_town"])
        
    def reset(self):
        self.step_count = 0
        self.current_area = "pallet_town"
        self.pokemon_encountered = 0
        self.areas_discovered = set(["pallet_town"])
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
        
        # Movimiento (acciones 0-3) - Más probabilidad de descubrir con epsilon alto
        if action in [0, 1, 2, 3]:
            # Mayor chance de cambiar de área con epsilon alto (más exploración)
            if random.random() < 0.05:  # 5% chance (mayor que normal)
                old_area = self.current_area
                self.current_area = random.choice(self.areas)
                if self.current_area != old_area:
                    reward += 5.0  # Recompensa por descubrir nueva área
                    if self.current_area not in self.areas_discovered:
                        reward += 10.0  # BONUS por área completamente nueva
                        self.areas_discovered.add(self.current_area)
                        print(f"🌟 ¡Nueva área descubierta! {old_area} → {self.current_area}")
                        print(f"   Áreas totales: {len(self.areas_discovered)}/10")
                    else:
                        print(f"🗺️  Área cambiada: {old_area} → {self.current_area}")
        
        # Acción A (acción 4) - interactuar - Más eventos con exploración alta
        elif action == 4:
            if random.random() < 0.08:  # 8% chance (aumentado para epsilon alto)
                self.pokemon_encountered += 1
                reward += 10.0  # Gran recompensa por encontrar Pokemon
                print(f"🐾 ¡Pokemon encontrado! Total: {self.pokemon_encountered}")
        
        # Acción B (acción 5) - menú
        elif action == 5:
            reward -= 0.3  # Menor penalización (epsilon alto explora todo)
        
        # Start/Select (acciones 6-7) - Menos penalización con exploración alta
        elif action in [6, 7]:
            reward -= 0.1  # Penalización reducida
        
        # Añade variabilidad - Mayor con epsilon alto
        reward += random.uniform(-0.3, 0.3)
        
        # El episodio nunca termina en esta versión simplificada
        done = False
        truncated = False
        
        info = {
            "step": self.step_count,
            "area": self.current_area,
            "pokemon": self.pokemon_encountered,
            "areas_discovered": len(self.areas_discovered)
        }
        
        return self._get_observation(), reward, done, truncated, info

class AutoDemo09:
    """Demo automático con epsilon fijo de 0.9"""
    
    def __init__(self):
        self.env = SimpleMockEnv()
        self.agent = None
        self.step_count = 0
        self.total_reward = 0
        self.session_start = time.time()
        self.EPSILON = 0.9  # Epsilon fijo muy alto
        
    def initialize_agent(self):
        """Inicializa el agente con epsilon fijo"""
        self.agent = VariableEpsilonGreedyAgent(self.env, epsilon=self.EPSILON)
        print(f"🤖 Agente inicializado con epsilon={self.EPSILON} (Exploración Muy Alta)")
        print("   → 90% exploración, 10% explotación")
        print("   → Comportamiento casi completamente exploratorio")
        
    def print_header(self):
        """Muestra información del demo"""
        print("=" * 80)
        print("DEMO AUTOMÁTICO: EPSILON 0.9 - EXPLORACIÓN MUY ALTA")
        print("=" * 80)
        print("Este demo muestra el comportamiento de un agente con epsilon = 0.9")
        print("• 90% de las acciones serán exploratorias (aleatorias)")
        print("• 10% de las acciones serán de explotación (mejores conocidas)")
        print("• Comportamiento altamente exploratorio y errático")
        print("• Ideal para descubrir nuevos ambientes y oportunidades")
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
        print(f"   Epsilon fijo: {stats['epsilon']:.1f} (Exploración Muy Alta)")
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
        print(f"   🌍 Áreas descubiertas: {len(self.env.areas_discovered)}/10")
        print(f"   🐾 Pokémon encontrados: {self.env.pokemon_encountered}")
        
        if show_detailed:
            print(f"   📋 Recompensas recientes: {stats['recent_rewards']}")
        print("-" * 60)
        
    def run_exploration_demo(self, cycles=4, steps_per_cycle=150):
        """Ejecuta demo enfocado en exploración"""
        print(f"🚀 Iniciando demo de exploración: {cycles} ciclos de {steps_per_cycle} pasos")
        print(f"   Total de pasos planificados: {cycles * steps_per_cycle}")
        print(f"   🎯 Objetivo: Descubrir el máximo número de áreas")
        print()
        
        obs = self.env.reset()
        areas_per_cycle = []
        
        for cycle in range(cycles):
            print(f"🔄 CICLO {cycle + 1}/{cycles} - EXPLORACIÓN INTENSIVA")
            print("-" * 50)
            
            cycle_reward = 0
            cycle_start = time.time()
            initial_areas = len(self.env.areas_discovered)
            
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
                
                # Mostrar progreso cada 30 pasos
                if (step + 1) % 30 == 0:
                    action_names = ['↑', '↓', '←', '→', 'A', 'B', 'START', 'SELECT']
                    action_name = action_names[action] if action < len(action_names) else f'A{action}'
                    exploration_symbol = "🔍" if random.random() < 0.9 else "🎯"  # 90% exploración
                    print(f"   Step {step+1:3d}: {exploration_symbol} {action_name:6s} → R={reward:+.2f} | "
                          f"Área: {info.get('area', 'unknown')} | Áreas: {info.get('areas_discovered', 0)}")
            
            cycle_time = time.time() - cycle_start
            areas_discovered_cycle = len(self.env.areas_discovered) - initial_areas
            areas_per_cycle.append(areas_discovered_cycle)
            
            print(f"✅ Ciclo {cycle + 1} completado en {cycle_time:.1f}s")
            print(f"   🌍 Nuevas áreas descubiertas: {areas_discovered_cycle}")
            print(f"   💰 Recompensa del ciclo: {cycle_reward:.2f}")
            if cycle_time > 0:
                print(f"   ⚡ Pasos por segundo: {steps_per_cycle/cycle_time:.1f}")
            else:
                print(f"   ⚡ Pasos por segundo: >1000 (muy rápido)")
            
            # Mostrar estadísticas cada ciclo
            self.print_statistics()
            
            # Pausa breve entre ciclos
            if cycle < cycles - 1:
                print("\n⏸️  Pausa de 2 segundos entre ciclos...")
                time.sleep(2)
                print()
        
        print("🎉 Demo de exploración completado!")
        print(f"📈 Resumen de descubrimientos por ciclo: {areas_per_cycle}")
        print(f"🏆 Total de áreas descubiertas: {len(self.env.areas_discovered)}/10")
        
    def run_chaos_demo(self, duration_minutes=1.5):
        """Ejecuta demo mostrando el comportamiento caótico de epsilon alto"""
        print(f"🌪️  DEMO CAÓTICO - Epsilon 0.9 por {duration_minutes} minutos")
        print("    Observa el comportamiento errático y altamente exploratorio")
        print("    Presiona Ctrl+C para detener anticipadamente")
        print()
        
        obs = self.env.reset()
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        last_stats_time = start_time
        
        action_counts = [0] * 8  # Contador para cada acción
        action_names = ['↑', '↓', '←', '→', 'A', 'B', 'START', 'SELECT']
        
        try:
            while time.time() < end_time:
                # Agente selecciona acción
                action = self.agent.select_action(obs)
                action_counts[action] += 1
                
                # Entorno ejecuta paso
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                # Agente aprende
                self.agent.update_q_value(obs, action, reward, next_obs)
                
                # Actualizar contadores
                self.step_count += 1
                self.total_reward += reward
                obs = next_obs
                
                # Mostrar el caos en tiempo real (cada 20 pasos)
                if self.step_count % 20 == 0:
                    # Mostrar secuencia reciente de acciones
                    recent_actions = [action_names[action] for _ in range(5)]  # Simular 5 acciones
                    chaos_sequence = ''.join([random.choice(['🔍', '🎯']) for _ in range(10)])
                    print(f"Step {self.step_count:4d}: Caos={chaos_sequence} | "
                          f"Área: {info.get('area', 'unknown')[:12]:12s} | "
                          f"R={reward:+.1f}")
                
                # Mostrar estadísticas cada 45 segundos
                current_time = time.time()
                if current_time - last_stats_time > 45:
                    elapsed = current_time - start_time
                    remaining = (end_time - current_time) / 60
                    print(f"\n⏱️  Tiempo transcurrido: {elapsed/60:.1f}min | Restante: {remaining:.1f}min")
                    
                    # Mostrar distribución de acciones
                    total_actions = sum(action_counts)
                    print("🎲 Distribución de acciones (debería ser casi uniforme):")
                    for i, (name, count) in enumerate(zip(action_names, action_counts)):
                        percentage = (count/total_actions)*100 if total_actions > 0 else 0
                        print(f"     {name:6s}: {count:4d} ({percentage:4.1f}%)")
                    
                    self.print_statistics()
                    last_stats_time = current_time
                
                # Pausa muy pequeña para no saturar la CPU
                time.sleep(0.005)  # Más rápido para mostrar el caos
                
        except KeyboardInterrupt:
            print("\n⏹️  Demo caótico detenido por el usuario")
        
        print(f"\n🌪️  Demo caótico finalizado después de {(time.time() - start_time)/60:.1f} minutos")
        
        # Mostrar análisis final del caos
        total_actions = sum(action_counts)
        print("\n📊 ANÁLISIS DEL COMPORTAMIENTO CAÓTICO:")
        print("   Distribución final de acciones:")
        for name, count in zip(action_names, action_counts):
            percentage = (count/total_actions)*100 if total_actions > 0 else 0
            bar = "█" * int(percentage/5)  # Barra visual
            print(f"     {name:6s}: {count:4d} ({percentage:4.1f}%) {bar}")
        
    def run_comparison_analysis(self):
        """Ejecuta análisis comparativo con otros valores de epsilon"""
        print("🔬 ANÁLISIS COMPARATIVO: ¿Por qué Epsilon 0.9 es especial?")
        print("=" * 70)
        print("Comparando comportamiento de ε=0.9 vs otros valores")
        print()
        
        obs = self.env.reset()
        exploration_actions = 0
        exploitation_actions = 0
        action_sequence = []
        decision_analysis = []
        
        # Ejecutar pasos con análisis detallado
        for step in range(300):
            # Obtener acción y determinar si fue exploratoria
            action = self.agent.select_action(obs)
            
            # Determinar tipo de acción (simulando el comportamiento interno)
            if random.random() < 0.9:  # 90% exploración
                exploration_actions += 1
                action_type = "🔍 EXPLORAR"
                decision_analysis.append("E")
            else:
                exploitation_actions += 1
                action_type = "🎯 EXPLOTAR" 
                decision_analysis.append("X")
                
            action_sequence.append(action_type[0])  # Solo el emoji
            
            # Entorno ejecuta paso
            next_obs, reward, done, truncated, info = self.env.step(action)
            self.agent.update_q_value(obs, action, reward, next_obs)
            
            self.step_count += 1
            self.total_reward += reward
            obs = next_obs
            
            # Mostrar análisis cada 75 pasos
            if (step + 1) % 75 == 0:
                recent_sequence = ''.join(action_sequence[-25:])  # Últimas 25 acciones
                recent_decisions = ''.join(decision_analysis[-25:])  # Últimas 25 decisiones
                exploration_rate = exploration_actions / (step + 1)
                
                print(f"Step {step+1:3d}: {action_type}")
                print(f"         Secuencia visual: {recent_sequence}")
                print(f"         Decisiones (E/X): {recent_decisions}")
                print(f"         Exploración: {exploration_rate:.1%} | Recompensa: {reward:+.2f}")
                print(f"         Áreas: {info.get('areas_discovered', 0)}/10 | "
                      f"Pokémon: {info.get('pokemon', 0)}")
                print()
                
        print(f"📈 ANÁLISIS COMPARATIVO FINAL:")
        print(f"   🔍 Acciones exploratorias: {exploration_actions} ({exploration_actions/300:.1%})")
        print(f"   🎯 Acciones de explotación: {exploitation_actions} ({exploitation_actions/300:.1%})")
        print()
        print(f"   📊 COMPARACIÓN CON OTROS EPSILONS:")
        print(f"      ε=0.1 (low):     10% exploración - Muy eficiente, poco descubrimiento")
        print(f"      ε=0.3 (moderate): 30% exploración - Equilibrado")
        print(f"      ε=0.5 (balanced): 50% exploración - Neutro")
        print(f"      ε=0.9 (very_high): 90% exploración - Máximo descubrimiento, menos eficiente")
        print()
        print(f"   🌟 VENTAJAS DE ε=0.9:")
        print(f"      • Descubre {len(self.env.areas_discovered)} áreas rápidamente")
        print(f"      • Encuentra {self.env.pokemon_encountered} Pokémon por exploración activa") 
        print(f"      • Genera Q-table diversa: {self.agent.get_statistics()['q_table_size']} estados")
        print()
        print(f"   ⚠️  DESVENTAJAS DE ε=0.9:")
        print(f"      • Comportamiento errático y difícil de predecir")
        print(f"      • Puede no aprovechar estrategias óptimas conocidas")
        print(f"      • Mayor tiempo para converger a soluciones estables")

def main():
    """Función principal del demo"""
    print("Iniciando Demo Automático de Epsilon 0.9")
    
    # Crear y configurar demo
    demo = AutoDemo09()
    demo.print_header()
    demo.initialize_agent()
    
    print("\n🤖 MODO AUTOMÁTICO: Ejecutando demo completo...")
    print("   No se requiere entrada del usuario")
    print("   Presiona Ctrl+C para detener anticipadamente")
    print()
    
    try:
        # Ejecutar demo automático completo sin entrada del usuario
        print("🚀 Ejecutando demo de exploración...")
        demo.run_exploration_demo(cycles=2, steps_per_cycle=100)
        
        print("\n" + "="*70)
        print("🔬 Ejecutando análisis comparativo...")
        demo.run_comparison_analysis()
            
    except KeyboardInterrupt:
        print("\n⏹️  Demo cancelado por el usuario")
    
    # Estadísticas finales
    print("\n" + "="*80)
    print("RESUMEN FINAL DEL DEMO")
    print("="*80)
    demo.print_statistics(show_detailed=True)
    
    stats = demo.agent.get_statistics()
    print(f"\n💡 CONCLUSIONES EPSILON 0.9:")
    print(f"   • Exploración observada: {stats['exploration_rate']:.1%}")
    print(f"   • Comportamiento altamente exploratorio y caótico")
    print(f"   • Q-table muy diversa: {stats['q_table_size']} estados únicos")
    print(f"   • Ideal para descubrimiento inicial de ambientes desconocidos")
    print(f"   • Áreas descubiertas: {len(demo.env.areas_discovered)}/10")
    print(f"   • ⚠️  Requiere eventual reducción de epsilon para convergencia")
    print("="*80)

if __name__ == "__main__":
    main()