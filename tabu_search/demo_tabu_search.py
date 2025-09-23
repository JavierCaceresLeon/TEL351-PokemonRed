"""
Ejemplo de Uso del Agente Tabu Search
====================================

Este script demuestra cómo usar el nuevo agente Tabu Search
y comparar su comportamiento con otros agentes.
"""

from search_algorithms.tabu_agent import TabuSearchAgent, GameScenario
import numpy as np

def demo_tabu_search_features():
    """Demostrar las características únicas del Tabu Search"""
    print("🔍 Demostración del Agente Tabu Search")
    print("=" * 50)
    
    # Crear agente con configuraciones diferentes
    print("\n1. Creando agente Tabu Search...")
    agent = TabuSearchAgent(
        tabu_tenure=7,      # Mantener movimientos en lista tabú por 7 iteraciones
        max_tabu_size=50,   # Máximo 50 movimientos en memoria
        aspiration_threshold=1.5,  # Permitir movimientos tabú si calidad > 1.5x mejor
        scenario_detection_enabled=True
    )
    
    print(f"   ✅ Configuración:")
    print(f"      - Tenure Tabú: {agent.tabu_tenure} iteraciones")
    print(f"      - Tamaño Máximo: {agent.max_tabu_size} movimientos")
    print(f"      - Umbral Aspiración: {agent.aspiration_threshold:.1f}x")
    
    # Simular estados del juego
    print("\n2. Simulando decisiones del agente...")
    
    # Estado simulado
    mock_observation = np.random.rand(144, 160, 3)  # Imagen del Game Boy
    mock_game_state = {
        'hp': 85,
        'max_hp': 100,
        'level': 5,
        'badges': 0,
        'pcount': 0,
        'x': 10,
        'y': 15
    }
    
    # Realizar varias decisiones para demostrar la memoria tabú
    for i in range(10):
        action, decision_info = agent.select_action(mock_observation, mock_game_state)
        
        # Simular recompensa
        reward = np.random.uniform(-0.1, 0.5)
        agent.update_performance(action, reward, mock_observation, mock_game_state)
        
        print(f"   Paso {i+1}: Acción {action} | "
              f"Escenario: {decision_info['scenario']} | "
              f"Lista Tabú: {decision_info['tabu_list_size']} | "
              f"Calidad: {decision_info['selected_quality']:.3f}")
    
    # Mostrar métricas de exploración
    print("\n3. Métricas de exploración:")
    metrics = agent.get_exploration_metrics()
    print(f"   📍 Posiciones únicas visitadas: {metrics['unique_positions_visited']}")
    print(f"   🚶 Visitas totales: {metrics['total_position_visits']}")
    print(f"   📊 Eficiencia exploración: {metrics['exploration_efficiency']:.2%}")
    print(f"   🎯 Mejor calidad encontrada: {metrics['performance_metrics']['best_solution_quality']:.3f}")
    
    return agent

def compare_agents_behavior():
    """Comparar el comportamiento de diferentes agentes"""
    print("\n🆚 Comparación de Comportamiento de Agentes")
    print("=" * 50)
    
    # Crear agentes con diferentes configuraciones
    agents = {
        "Conservador": TabuSearchAgent(tabu_tenure=10, max_tabu_size=30),
        "Agresivo": TabuSearchAgent(tabu_tenure=5, max_tabu_size=70),
        "Aspiracional": TabuSearchAgent(tabu_tenure=7, aspiration_threshold=1.2)
    }
    
    mock_observation = np.random.rand(144, 160, 3)
    mock_game_state = {'hp': 100, 'max_hp': 100, 'level': 1, 'badges': 0, 'pcount': 0}
    
    print("\nComportamiento después de 20 pasos:")
    for name, agent in agents.items():
        for _ in range(20):
            action, _ = agent.select_action(mock_observation, mock_game_state)
            agent.update_performance(action, np.random.uniform(0, 1), mock_observation, mock_game_state)
        
        metrics = agent.get_exploration_metrics()
        print(f"   {name}:")
        print(f"      - Lista Tabú: {len(agent.tabu_list)} movimientos")
        print(f"      - Posiciones únicas: {metrics['unique_positions_visited']}")
        print(f"      - Mejor calidad: {metrics['performance_metrics']['best_solution_quality']:.3f}")

def usage_recommendations():
    """Recomendaciones de uso para diferentes situaciones"""
    print("\n💡 Recomendaciones de Uso")
    print("=" * 50)
    
    recommendations = [
        {
            "Situación": "Exploración inicial (comenzar el juego)",
            "Configuración": "tabu_tenure=5, max_tabu_size=30",
            "Razón": "Permite más libertad para descubrir el mapa"
        },
        {
            "Situación": "Navegación específica (ir a un lugar)",
            "Configuración": "tabu_tenure=10, aspiration_threshold=1.2",
            "Razón": "Evita volver atrás pero permite shortcuts"
        },
        {
            "Situación": "Combates repetitivos",
            "Configuración": "tabu_tenure=7, max_tabu_size=50",
            "Razón": "Balance entre memoria y flexibilidad"
        },
        {
            "Situación": "Exploración avanzada (mapas complejos)",
            "Configuración": "tabu_tenure=12, max_tabu_size=70",
            "Razón": "Memoria extendida para evitar bucles largos"
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['Situación']}")
        print(f"   ⚙️  {rec['Configuración']}")
        print(f"   📝 {rec['Razón']}")

def main():
    """Función principal de demostración"""
    print("🎮 Demo Completo del Agente Tabu Search para Pokemon Red")
    print("🔗 Ejecuta este script para entender las características únicas")
    print()
    
    # Ejecutar demostraciones
    agent = demo_tabu_search_features()
    compare_agents_behavior()
    usage_recommendations()
    
    print(f"\n📚 Próximos Pasos:")
    print(f"   1. Ejecuta: python run_tabu_interactive_metrics.py")
    print(f"   2. Déjalo correr un rato y para con Ctrl+C")
    print(f"   3. Revisa: results/tabu_search_metrics_*.md")
    print(f"   4. Ejecuta: python generate_metrics_visualizations.py")
    print(f"   5. Compara con Epsilon Greedy y PPO")
    
    print(f"\n🔍 Características Únicas del Tabu Search:")
    print(f"   ✅ Evita ciclos y comportamiento repetitivo")
    print(f"   ✅ Memoria de estados para navegación inteligente")
    print(f"   ✅ Criterios de aspiración para movimientos excepcionales")
    print(f"   ✅ Detección automática de atascamiento")
    print(f"   ✅ Compatible con sistema de métricas unificado")

if __name__ == "__main__":
    main()