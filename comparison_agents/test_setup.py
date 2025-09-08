"""
Script de prueba rápida para verificar que los agentes funcionan
Ejecuta una prueba básica de cada componente
"""

import sys
import os
from pathlib import Path

# Agregar directorios al path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_imports():
    """Probar que se pueden importar todos los módulos"""
    print("=== Probando Imports ===")
    
    try:
        from config import QuickTestConfig
        print("✓ Config importado correctamente")
    except ImportError as e:
        print(f"✗ Error importando config: {e}")
        return False
    
    try:
        from search_env import SearchEnvironment
        print("✓ SearchEnvironment importado correctamente")
    except ImportError as e:
        print(f"✗ Error importando SearchEnvironment: {e}")
        return False
    
    try:
        from search_algorithms.astar_agent import AStarAgent
        print("✓ AStarAgent importado correctamente")
    except ImportError as e:
        print(f"✗ Error importando AStarAgent: {e}")
        return False
    
    try:
        from search_algorithms.tabu_agent import TabuSearchAgent
        print("✓ TabuSearchAgent importado correctamente")
    except ImportError as e:
        print(f"✗ Error importando TabuSearchAgent: {e}")
        return False
    
    try:
        from v2_agent import V2TrainedAgent
        print("✓ V2TrainedAgent importado correctamente")
    except ImportError as e:
        print(f"⚠ Error importando V2TrainedAgent (normal si no hay stable_baselines3): {e}")
    
    return True

def test_config():
    """Probar configuración"""
    print("\n=== Probando Configuración ===")
    
    from config import QuickTestConfig
    
    # Validar configuración
    errors = QuickTestConfig.validate_config()
    
    if errors:
        print("✗ Errores de configuración encontrados:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✓ Configuración válida")
        return True

def test_search_environment():
    """Probar entorno de búsqueda básico"""
    print("\n=== Probando Entorno de Búsqueda ===")
    
    try:
        from config import QuickTestConfig
        from search_env import SearchEnvironment
        
        config = QuickTestConfig.get_search_env_config()
        print(f"Configuración: {config}")
        
        # Intentar crear entorno
        env = SearchEnvironment(config)
        print("✓ Entorno creado correctamente")
        
        # Probar reset
        state = env.reset()
        print(f"✓ Reset exitoso, posición inicial: {state['position']}")
        
        # Probar algunas acciones
        for i in range(3):
            actions = env.get_valid_actions(state)
            if actions:
                action = actions[0]
                state, reward, done = env.step(action)
                print(f"✓ Paso {i+1}: acción {action}, recompensa {reward:.3f}, terminado: {done}")
                
                if done:
                    break
            else:
                print("✗ No hay acciones válidas")
                break
        
        env.close()
        print("✓ Entorno cerrado correctamente")
        return True
        
    except Exception as e:
        print(f"✗ Error probando entorno: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_astar_basic():
    """Probar A* básico"""
    print("\n=== Probando A* Básico ===")
    
    try:
        from config import QuickTestConfig
        from search_env import SearchEnvironment
        from search_algorithms.astar_agent import AStarAgent
        
        config = QuickTestConfig.get_search_env_config()
        env = SearchEnvironment(config)
        
        # Crear agente con límites muy bajos para prueba rápida
        agent = AStarAgent(env, max_search_depth=10)
        print("✓ Agente A* creado")
        
        # Intentar búsqueda rápida
        plan = agent.search()
        print(f"✓ Búsqueda completada, plan de {len(plan)} acciones")
        
        # Obtener estadísticas
        stats = agent.get_stats()
        print(f"✓ Estadísticas: {stats}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error probando A*: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tabu_basic():
    """Probar Tabú Search básico"""
    print("\n=== Probando Tabú Search Básico ===")
    
    try:
        from config import QuickTestConfig
        from search_env import SearchEnvironment
        from search_algorithms.tabu_agent import TabuSearchAgent
        
        config = QuickTestConfig.get_search_env_config()
        env = SearchEnvironment(config)
        
        # Crear agente con límites muy bajos para prueba rápida
        agent = TabuSearchAgent(env, max_iterations=5, tabu_size=3)
        print("✓ Agente Tabú creado")
        
        # Intentar búsqueda rápida
        plan = agent.search()
        print(f"✓ Búsqueda completada, plan de {len(plan)} acciones")
        
        # Obtener estadísticas
        stats = agent.get_stats()
        print(f"✓ Estadísticas: {stats}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error probando Tabú Search: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ejecutar todas las pruebas"""
    print("Script de Prueba Rápida - Comparación de Agentes")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuración", test_config),
        ("Entorno de Búsqueda", test_search_environment),
        ("A* Básico", test_astar_basic),
        ("Tabú Search Básico", test_tabu_basic),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Error crítico en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen
    print("\n" + "=" * 50)
    print("RESUMEN DE PRUEBAS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{total} pruebas pasadas")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! El sistema está listo.")
        print("\nPara ejecutar la comparación completa:")
        print("  python run_comparison.py")
    else:
        print("⚠️ Algunas pruebas fallaron. Revisa los errores arriba.")
        print("\nPasos sugeridos:")
        print("1. Verifica que todos los archivos requeridos existen")
        print("2. Instala las dependencias faltantes")
        print("3. Ejecuta este script nuevamente")

if __name__ == "__main__":
    main()
