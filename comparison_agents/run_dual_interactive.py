"""
Ejecutor Simultáneo: Epsilon Greedy vs PPO Interactive
=====================================================

Este script ejecuta simultáneamente dos agentes con ventanas separadas del Game Boy:
1. Agente Epsilon Greedy (comparison_agents)
2. Agente PPO preentrenado (v2)

Permite comparar visualmente el comportamiento de ambos algoritmos en tiempo real.
"""

import subprocess
import sys
import time
from pathlib import Path
import threading

def run_epsilon_greedy():
    """Ejecutar agente Epsilon Greedy"""
    print("🎮 Iniciando Agente Epsilon Greedy...")
    try:
        subprocess.run([
            sys.executable, 
            "run_epsilon_greedy_interactive.py"
        ], cwd=Path(__file__).parent, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando Epsilon Greedy: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Epsilon Greedy interrumpido por usuario")

def run_ppo_agent():
    """Ejecutar agente PPO"""
    print("🤖 Iniciando Agente PPO...")
    try:
        subprocess.run([
            sys.executable, 
            "run_pretrained_interactive.py"
        ], cwd=Path(__file__).parent.parent / "v2", check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando PPO: {e}")
    except KeyboardInterrupt:
        print("\n🛑 PPO interrumpido por usuario")

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║              COMPARACIÓN SIMULTÁNEA DE AGENTES                   ║
║                 Epsilon Greedy vs PPO Interactive                ║
╚═══════════════════════════════════════════════════════════════════╝

📋 Instrucciones:
   • Se abrirán 2 ventanas del Game Boy simultáneamente
   • Ventana 1: Agente Epsilon Greedy (Heurístico)
   • Ventana 2: Agente PPO (Deep Learning)
   • Usa Ctrl+C para detener ambos agentes

⚠️  Requisitos:
   • Ambiente conda activado (pokemon-red-comparison)
   • Archivos events.json y map_data.json en ambos directorios
   • Modelo PPO entrenado en v2/runs/

🚀 Iniciando ejecución simultánea...
""")
    
    time.sleep(2)
    
    try:
        # Crear hilos para ejecutar ambos agentes simultáneamente
        thread_epsilon = threading.Thread(target=run_epsilon_greedy, name="EpsilonGreedy")
        thread_ppo = threading.Thread(target=run_ppo_agent, name="PPO")
        
        # Iniciar ambos hilos
        thread_epsilon.start()
        time.sleep(1)  # Pequeña pausa para evitar conflictos de inicio
        thread_ppo.start()
        
        print("✅ Ambos agentes iniciados correctamente.")
        print("💡 Presiona Ctrl+C para detener la ejecución simultánea.")
        
        # Esperar a que terminen ambos hilos
        thread_epsilon.join()
        thread_ppo.join()
        
        print("\n🏁 Ejecución simultánea completada.")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Deteniendo ejecución simultánea...")
        print("⏳ Esperando que terminen los procesos...")
        time.sleep(2)
        print("✅ Procesos finalizados.")
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
    
    print("\n📊 Revisa las métricas guardadas en:")
    print("   • comparison_agents/results/ (Epsilon Greedy)")
    print("   • v2/session_*/ (PPO)")