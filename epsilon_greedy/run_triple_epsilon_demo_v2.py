"""
Triple Demo de Epsilon Greedy - Versión Mejorada
===============================================

Versión mejorada del runner triple que ejecuta simultáneamente tres demos de epsilon greedy:
- Epsilon 0.3 (exploración moderada)
- Epsilon 0.9 (exploración caótica) 
- Epsilon interactivo (variable)

MEJORAS EN ESTA VERSIÓN:
- 🏷️ Títulos de ventana identificativos para cada demo
- 🤝 Inicio coordinado para evitar crashes
- 🛡️ Mejor manejo de errores y estabilidad
- 📊 Métricas completas guardadas automáticamente

Permite comparar visualmente el comportamiento de los tres valores de epsilon en tiempo real.
"""

import subprocess
import sys
import time
from pathlib import Path
import threading
import signal
import os

class TripleEpsilonRunnerV2:
    def __init__(self):
        self.processes = []
        self.threads = []
        self.running = True
        self.start_event = threading.Event()  # Evento para sincronizar inicio
        self.ready_count = 0
        self.ready_lock = threading.Lock()
        
    def wait_for_start_signal(self, demo_name):
        """Esperar señal de inicio coordinado"""
        print(f"🔄 {demo_name} esperando señal de inicio...")
        self.start_event.wait()  # Esperar hasta que se dé la señal
        print(f"🚀 {demo_name} iniciando ejecución!")
        
    def mark_ready(self, demo_name):
        """Marcar demo como lista y verificar si todas están listas"""
        with self.ready_lock:
            self.ready_count += 1
            print(f"✅ {demo_name} está listo ({self.ready_count}/3)")
            
            if self.ready_count == 3:
                print("\n🎯 ¡TODAS LAS DEMOS ESTÁN LISTAS!")
                print("⏳ Esperando 3 segundos antes del inicio coordinado...")
                time.sleep(3)
                print("🚀 ¡INICIANDO TODAS LAS DEMOS SIMULTÁNEAMENTE!")
                self.start_event.set()  # Dar señal para que todas inicien
        
    def run_demo(self, script_name, demo_name):
        """Ejecutar un demo específico con coordinación de inicio"""
        try:
            # Marcar como listo y esperar inicio coordinado
            self.mark_ready(demo_name)
            self.wait_for_start_signal(demo_name)
            
            script_path = Path(__file__).parent / script_name
            
            if not script_path.exists():
                print(f"❌ Error: {script_name} no encontrado")
                return
            
            env = os.environ.copy()
            env['DEMO_WINDOW_TITLE'] = demo_name  # Variable de entorno para el título
            
            print(f"🎮 Iniciando {demo_name}...")
            
            # Usar subprocess.run para mejor control
            result = subprocess.run(
                [sys.executable, str(script_path)],
                env=env,
                capture_output=False,  # Mostrar output en tiempo real
                text=True,
                cwd=Path(__file__).parent
            )
            
            if result.returncode != 0:
                print(f"⚠️ {demo_name} terminó con código de salida: {result.returncode}")
            else:
                print(f"✅ {demo_name} completado exitosamente")
                
        except Exception as e:
            print(f"❌ Error ejecutando {demo_name}: {e}")
        finally:
            print(f"🔄 {demo_name} finalizó")

    def signal_handler(self, signum, frame):
        """Manejador de señales para Ctrl+C"""
        print(f"\n🛑 Señal {signum} recibida. Deteniendo todas las demos...")
        self.stop_all()

    def stop_all(self):
        """Detener todos los procesos"""
        self.running = False
        print("⏳ Deteniendo procesos...")
        
        for i, process in enumerate(self.processes):
            if process.poll() is None:  # Si el proceso sigue ejecutándose
                try:
                    print(f"🛑 Deteniendo proceso {i+1}...")
                    process.terminate()
                    # Esperar 3 segundos para terminación suave
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        print(f"⚡ Forzando cierre del proceso {i+1}...")
                        process.kill()
                        process.wait()
                except Exception as e:
                    print(f"❌ Error deteniendo proceso {i+1}: {e}")
        
        print("✅ Todos los procesos detenidos.")

    def run_triple_demo(self):
        """Ejecutar los tres demos simultáneamente con coordinación mejorada"""
        print("🚀 TRIPLE DEMO DE EPSILON GREEDY - VERSION MEJORADA")
        print("=" * 65)
        
        print("\n📋 CONFIGURACIÓN:")
        print("  🎯 Demo 1: Epsilon 0.3 (Exploración Moderada)")
        print("  🌪️ Demo 2: Epsilon 0.9 (Exploración Caótica)")
        print("  🎮 Demo 3: Epsilon Interactivo (Variable)")
        
        print("\n✨ MEJORAS EN ESTA VERSIÓN:")
        print("  🏷️ Cada ventana PyBoy tendrá un título identificativo")
        print("  🤝 Inicio coordinado para evitar crashes")
        print("  🛡️ Mejor manejo de errores y estabilidad")
        print("  📊 Métricas completas guardadas automáticamente")
        
        print("\n⚠️  IMPORTANTE:")
        print("  - Todas las demos iniciarán de forma coordinada")
        print("  - Busca 3 ventanas PyBoy con títulos diferentes") 
        print("  - Usa Ctrl+C para detener todas simultáneamente")
        print("  - Los resultados se guardan en epsilon_greedy/results/")
        
        # Configurar signal handler para Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Input del usuario para continuar
        print("\n⏳ PREPARACIÓN:")
        try:
            input("👤 Presiona ENTER para iniciar las tres demos simultáneamente...")
        except KeyboardInterrupt:
            print("\n🛑 Cancelado por el usuario")
            return
        
        print("\n🔄 INICIANDO SETUP COORDINADO...")
        
        # Crear threads para cada demo
        demos = [
            ("demo_pyboy_epsilon_03.py", "Pokemon Red - Epsilon 0.3 (Moderado)"),
            ("demo_pyboy_epsilon_09.py", "Pokemon Red - Epsilon 0.9 (Caótico)"), 
            ("run_epsilon_greedy_interactive.py", "Pokemon Red - Epsilon Interactivo")
        ]
        
        threads = []
        for script_name, demo_name in demos:
            thread = threading.Thread(
                target=self.run_demo,
                args=(script_name, demo_name),
                daemon=True
            )
            threads.append(thread)
            print(f"  ✅ Thread preparado para: {demo_name}")
        
        print(f"\n🎬 Iniciando {len(threads)} demos coordinadamente...")
        
        # Iniciar todos los threads
        for i, thread in enumerate(threads):
            thread.start()
            print(f"  🚀 Demo {i+1} iniciado")
            time.sleep(1)  # Pequeño delay entre inicios
        
        print("\n🎮 ¡DEMOS EN EJECUCIÓN!")
        print("📺 Busca las 3 ventanas PyBoy con títulos identificativos:")
        print("  🎯 'Pokemon Red - Epsilon 0.3 (Moderado)'")
        print("  🌪️ 'Pokemon Red - Epsilon 0.9 (Caótico)'")  
        print("  🎮 'Pokemon Red - Epsilon Interactivo'")
        print("\n🛑 Presiona Ctrl+C aquí para detener todas las demos")
        
        try:
            # Esperar a que terminen todos los threads
            for i, thread in enumerate(threads):
                if thread.is_alive():
                    print(f"⏳ Esperando finalización de demo {i+1}...")
                    thread.join()
                    print(f"✅ Demo {i+1} completado")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupción detectada, deteniendo demos...")
            self.stop_all()
        
        print("\n🎉 ¡TODAS LAS DEMOS HAN FINALIZADO!")
        print("📊 Revisa la carpeta 'epsilon_greedy/results/' para las métricas")
        print("💡 Compara los diferentes comportamientos de epsilon observados")
        
        print("\n📈 ANÁLISIS SUGERIDO:")
        print("  1. Compara las tasas de exploración vs explotación")
        print("  2. Observa las diferencias en recompensas obtenidas")
        print("  3. Analiza qué epsilon fue más eficiente")
        print("  4. Revisa los patrones de movimiento en los historiales")

def main():
    """Función principal"""
    runner = TripleEpsilonRunnerV2()
    runner.run_triple_demo()

if __name__ == "__main__":
    main()