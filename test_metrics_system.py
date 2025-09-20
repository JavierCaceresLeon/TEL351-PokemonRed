"""
Script de Prueba del Sistema de Métricas
=======================================

Este script permite probar rápidamente el sistema de métricas mejorado
y verificar que todo funciona correctamente.
"""

import os
import subprocess
import time
from pathlib import Path

def test_epsilon_greedy_metrics():
    """Probar las métricas del agente Epsilon Greedy"""
    print("🤖 Probando métricas de Epsilon Greedy...")
    print("   (Se ejecutará por 30 segundos, luego se detendrá automáticamente)")
    
    # Cambiar al directorio correcto
    os.chdir("comparison_agents")
    
    try:
        # Ejecutar el script por un tiempo limitado
        process = subprocess.Popen(["python", "run_epsilon_greedy_interactive.py"])
        time.sleep(30)  # Ejecutar por 30 segundos
        process.terminate()
        process.wait()
        
        # Verificar que se generaron archivos
        results_dir = Path("results")
        if results_dir.exists():
            files = list(results_dir.glob("epsilon_greedy_*"))
            print(f"   ✅ Generados {len(files)} archivos de métricas")
            for file in files[-3:]:  # Mostrar los últimos 3
                print(f"      📄 {file.name}")
        else:
            print("   ❌ No se encontraron archivos de métricas")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    os.chdir("..")

def test_ppo_metrics():
    """Probar las métricas del agente PPO"""
    print("\n🧠 Probando métricas de PPO...")
    print("   (Se ejecutará por 30 segundos, luego se detendrá automáticamente)")
    
    # Cambiar al directorio v2
    os.chdir("v2")
    
    try:
        # Verificar que existe el modelo
        runs_dir = Path("runs")
        if not runs_dir.exists() or not list(runs_dir.glob("*.zip")):
            print("   ⚠️ No se encontraron modelos PPO en v2/runs/")
            print("   📋 Para probar PPO, necesitas un modelo preentrenado")
            os.chdir("..")
            return
        
        # Ejecutar el script por un tiempo limitado
        process = subprocess.Popen(["python", "run_ppo_interactive_metrics.py"])
        time.sleep(30)  # Ejecutar por 30 segundos
        process.terminate()
        process.wait()
        
        # Verificar que se generaron archivos
        results_dir = Path("ppo_results")
        if results_dir.exists():
            files = list(results_dir.glob("ppo_*"))
            print(f"   ✅ Generados {len(files)} archivos de métricas")
            for file in files[-3:]:  # Mostrar los últimos 3
                print(f"      📄 {file.name}")
        else:
            print("   ❌ No se encontraron archivos de métricas")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    os.chdir("..")

def test_visualizations():
    """Probar el generador de visualizaciones"""
    print("\n🎨 Probando generador de visualizaciones...")
    
    os.chdir("comparison_agents")
    
    try:
        # Ejecutar el generador de visualizaciones
        result = subprocess.run(["python", "generate_metrics_visualizations.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("   ✅ Visualizaciones generadas exitosamente")
            
            # Verificar archivos generados
            viz_dir = Path("visualization_output")
            if viz_dir.exists():
                files = list(viz_dir.glob("*"))
                print(f"   📊 Generados {len(files)} archivos de visualización")
                for file in files[-3:]:
                    print(f"      🎯 {file.name}")
        else:
            print(f"   ❌ Error en visualizaciones: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("   ⏰ Timeout - Las visualizaciones pueden tomar tiempo con muchos datos")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    os.chdir("..")

def show_directory_structure():
    """Mostrar la estructura de directorios de métricas"""
    print("\n📁 Estructura de directorios de métricas:")
    
    directories = [
        "comparison_agents/results",
        "v2/ppo_results", 
        "comparison_agents/visualization_output"
    ]
    
    for dir_path in directories:
        path = Path(dir_path)
        if path.exists():
            files = list(path.glob("*"))
            print(f"   📂 {dir_path}/ ({len(files)} archivos)")
            for file in files[-2:]:  # Mostrar últimos 2
                print(f"      📄 {file.name}")
        else:
            print(f"   📂 {dir_path}/ (no existe)")

def main():
    """Función principal de prueba"""
    print("🔬 PRUEBA DEL SISTEMA DE MÉTRICAS COMPLETO")
    print("=" * 50)
    
    # Verificar que estamos en el directorio correcto
    if not Path("comparison_agents").exists():
        print("❌ Ejecuta este script desde el directorio TEL351-PokemonRed/")
        return
    
    print("📋 Esta prueba:")
    print("   1. Ejecutará Epsilon Greedy por 30 segundos")
    print("   2. Intentará ejecutar PPO por 30 segundos (si hay modelo)")
    print("   3. Generará visualizaciones de los datos existentes")
    print("   4. Mostrará la estructura de archivos creada")
    
    input("\n⏸️  Presiona Enter para continuar...")
    
    # Ejecutar pruebas
    test_epsilon_greedy_metrics()
    test_ppo_metrics()
    test_visualizations()
    show_directory_structure()
    
    print("\n✅ PRUEBAS COMPLETADAS")
    print("\n📚 Para uso normal:")
    print("   - Ejecuta run_epsilon_greedy_interactive.py y para con Ctrl+C")
    print("   - Ejecuta v2/run_ppo_interactive_metrics.py y para con Ctrl+C")
    print("   - Ejecuta generate_metrics_visualizations.py para gráficos")
    print("   - Revisa las carpetas results/ y visualization_output/")

if __name__ == "__main__":
    main()