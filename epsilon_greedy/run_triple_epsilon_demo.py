"""
Ejecutor Simultáneo: Triple Demo Epsilon Greedy
===============================================

Este script ejecuta simultáneamente tres agentes Epsilon Greedy con ventanas separadas del Game Boy:
1. Demo Epsilon 0.3 (Exploración Moderada) - demo_pyboy_epsilon_03.py
2. Demo Epsilon 0.9 (Exploración Caótica) - demo_pyboy_epsilon_09.py  
3. Demo Epsilon Variable (Interactivo) - run_epsilon_greedy_interactive.py

MEJORAS:
- ✅ Rótulos para identificar cada ventana
- ✅ Inicio manual coordinado para evitar caídas
- ✅ Espera a que todas las ventanas estén listas antes de comenzar

Permite comparar visualmente el comportamiento de los tres valores de epsilon en tiempo real.
"""

import subprocess
import sys
import time
from pathlib import Path
import threading
import signal
import os

class TripleEpsilonRunner:
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
            
            print(f"� Iniciando {demo_name}...")
            
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
        
    def run_epsilon_03(self):
        """Ejecutar demo con epsilon 0.3"""
        self.run_demo("demo_pyboy_epsilon_03.py", "Demo Epsilon 0.3 (Moderado)")

    def run_epsilon_09(self):
        """Ejecutar demo con epsilon 0.9"""
        self.run_demo("demo_pyboy_epsilon_09.py", "Demo Epsilon 0.9 (Caótico)")

    def run_epsilon_interactive(self):
        """Ejecutar demo epsilon variable (interactivo)"""
        self.run_demo("run_epsilon_greedy_interactive.py", "Demo Interactivo (Variable)")

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
                    print(f"⚠️ Error deteniendo proceso {i+1}: {e}")
        
        print("✅ Todos los procesos detenidos.")

    def run_triple_demo(self):
        """Ejecutar las tres demos simultáneamente"""
        print("""
╔═══════════════════════════════════════════════════════════════════╗
║                    TRIPLE DEMO EPSILON GREEDY                    ║
║          Comparación Simultánea de Tres Valores de Epsilon       ║
╚═══════════════════════════════════════════════════════════════════╝

📋 Configuración:
   🎯 Ventana 1: Epsilon 0.3 (30% exploración, 70% explotación)
   🌪️ Ventana 2: Epsilon 0.9 (90% exploración, 10% explotación) 
   🎮 Ventana 3: Epsilon Variable (Interactivo - puedes cambiar epsilon)

📊 Comparación Visual:
   • Epsilon 0.3: Comportamiento más estratégico y dirigido
   • Epsilon 0.9: Comportamiento caótico y muy exploratorio
   • Epsilon Variable: Comportamiento que puedes ajustar en tiempo real

⚠️  Requisitos:
   • Ambiente Python con todas las dependencias instaladas
   • Archivos init.state y PokemonRed.gb en directorio padre
   • Suficiente memoria para ejecutar 3 emuladores simultáneamente

💾 Métricas:
   • Cada demo guardará sus métricas automáticamente
   • Usa Ctrl+C para detener todas las demos y guardar métricas
   • Revisa epsilon_greedy/results/ después de la ejecución

🚀 Iniciando triple ejecución...
""")
        
        # Configurar manejador de señales
        signal.signal(signal.SIGINT, self.signal_handler)
        
        time.sleep(3)
        
        try:
            # Crear hilos para ejecutar las tres demos simultáneamente
            thread_03 = threading.Thread(target=self.run_epsilon_03, name="Epsilon-0.3")
            thread_09 = threading.Thread(target=self.run_epsilon_09, name="Epsilon-0.9") 
            thread_interactive = threading.Thread(target=self.run_epsilon_interactive, name="Epsilon-Interactive")
            
            self.threads = [thread_03, thread_09, thread_interactive]
            
            # Iniciar los tres hilos con pequeños delays para evitar conflictos
            print("🎯 Iniciando Epsilon 0.3...")
            thread_03.start()
            time.sleep(2)
            
            print("🌪️ Iniciando Epsilon 0.9...")
            thread_09.start()
            time.sleep(2)
            
            print("🎮 Iniciando Epsilon Interactivo...")
            thread_interactive.start()
            time.sleep(1)
            
            print("""
✅ Las tres demos han sido iniciadas correctamente.

🎮 CONTROLES DISPONIBLES:
   • Ventana Epsilon 0.3: Automática (sin controles)
   • Ventana Epsilon 0.9: Automática (sin controles)
   • Ventana Epsilon Interactivo: 
     - Teclas 1-7: Cambiar presets de epsilon
     - +/-: Ajustar epsilon manualmente
     - s: Ver estadísticas
     - q: Salir solo de esa ventana

🛑 DETENER TODO: Ctrl+C en esta terminal

📊 OBSERVA LAS DIFERENCIAS:
   • Epsilon 0.3: Movimientos más consistentes y dirigidos
   • Epsilon 0.9: Movimientos muy aleatorios y caóticos
   • Epsilon Interactivo: Comportamiento que puedes modificar

💡 TIP: Cambia el epsilon en la ventana interactiva para ver
   cómo se compara con las versiones fijas!
""")
            
            # Esperar a que terminen todos los hilos
            for thread in self.threads:
                thread.join()
            
            if self.running:
                print("\n🏁 Triple ejecución completada normalmente.")
            
        except KeyboardInterrupt:
            print("\n🛑 Ctrl+C detectado. Deteniendo triple ejecución...")
            self.stop_all()
        except Exception as e:
            print(f"\n❌ Error durante la ejecución: {e}")
            self.stop_all()
        
        print(f"""
🏁 TRIPLE DEMO FINALIZADA

📊 Revisa las métricas guardadas en:
   📁 epsilon_greedy/results/
   
📈 Archivos generados:
   • demo_epsilon_03_metrics_[timestamp].md
   • demo_epsilon_09_metrics_[timestamp].md  
   • epsilon_greedy_metrics_[timestamp].md (si usaste la demo interactiva)
   
🔍 Análisis sugerido:
   1. Compara las tasas de exploración vs explotación
   2. Observa las diferencias en recompensas obtenidas
   3. Analiza qué epsilon fue más eficiente para obtener Pokemon
   4. Revisa los patrones de movimiento en los historiales de acciones

💡 Conclusiones esperadas:
   • Epsilon 0.3: Balance óptimo entre exploración y explotación
   • Epsilon 0.9: Exploración extensiva pero posiblemente ineficiente
   • Epsilon Variable: Permite experimentar con diferentes estrategias
""")

def main():
    """Función principal"""
    runner = TripleEpsilonRunner()
    runner.run_triple_demo()

if __name__ == "__main__":
    main()