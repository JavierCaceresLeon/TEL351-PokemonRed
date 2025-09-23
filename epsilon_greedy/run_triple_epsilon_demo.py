"""
Ejecutor Simultáneo: Triple Demo Epsilon Greedy
===============================================

Este script ejecuta simultáneamente tres agentes Epsilon Greedy con ventanas separadas del Game Boy:
1. Demo Epsilon 0.3 (Exploración Moderada) - demo_pyboy_epsilon_03.py
2. Demo Epsilon 0.9 (Exploración Caótica) - demo_pyboy_epsilon_09.py  
3. Demo Epsilon Variable (Interactivo) - run_epsilon_greedy_interactive.py

MEJORAS INCLUIDAS:
-  Rótulos para identificar cada ventana del PyBoy  
-  Títulos de ventana súper largos y descriptivos
-  Prefijos únicos en TODAS las salidas de consola
-  Posicionamiento automático de ventanas en pantalla
-  Múltiples métodos de identificación robustos
-  Inicio manual coordinado para evitar caídas
-  Espera a que todas las ventanas estén listas antes de comenzar

IDENTIFICACIÓN VISUAL MEJORADA:
-  EPSILON 0.3: Título súper largo "POKEMON RED ===>>> EPSILON 0.3 MODERADO <<<==== 30-EXPLORA 70-EXPLOTA"
  + Posición: Esquina superior izquierda (100, 100)
  + Prefijo consola: [EPSILON-0.3-MODERADO]
  + Comportamiento: 30% exploración, 70% explotación
   
-  EPSILON 0.9: Título súper largo "POKEMON RED ===>>> EPSILON 0.9 CAOTICO <<<==== 90-EXPLORA 10-EXPLOTA"
  + Posición: Centro superior (500, 100)  
  + Prefijo consola: [EPSILON-0.9-CAOTICO]
  + Comportamiento: 90% exploración, 10% explotación
   
-  EPSILON VARIABLE: Título súper largo "POKEMON RED ===>>> EPSILON VARIABLE INTERACTIVO <<<==== ADAPTATIVO"
  + Posición: Esquina superior derecha (900, 100)
  + Prefijo consola: [EPSILON-VARIABLE-INTERACTIVO]
  + Comportamiento: Epsilon adaptativo según escenario

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
        print(f" {demo_name} esperando señal de inicio...")
        self.start_event.wait()  # Esperar hasta que se dé la señal
        print(f" {demo_name} iniciando ejecución!")
        
    def mark_ready(self, demo_name):
        """Marcar demo como lista y verificar si todas están listas"""
        with self.ready_lock:
            self.ready_count += 1
            print(f" {demo_name} está listo ({self.ready_count}/3)")
            
            if self.ready_count == 3:
                print("\n ¡TODAS LAS DEMOS ESTÁN LISTAS!")
                print(" Esperando 3 segundos antes del inicio coordinado...")
                time.sleep(3)
                print(" ¡INICIANDO TODAS LAS DEMOS SIMULTÁNEAMENTE!")
                self.start_event.set()  # Dar señal para que todas inicien
        
    def run_demo(self, script_name, demo_name):
        """Ejecutar un demo específico con coordinación de inicio"""
        try:
            # Marcar como listo y esperar inicio coordinado
            self.mark_ready(demo_name)
            self.wait_for_start_signal(demo_name)
            
            script_path = Path(__file__).parent / script_name
            
            if not script_path.exists():
                print(f" Error: {script_name} no encontrado")
                return
            
            env = os.environ.copy()
            env['DEMO_WINDOW_TITLE'] = demo_name  # Variable de entorno para el título
            
            print(f" Iniciando {demo_name}...")
            
            # Usar subprocess.run para mejor control
            result = subprocess.run(
                [sys.executable, str(script_path)],
                env=env,
                capture_output=False,  # Mostrar output en tiempo real
                text=True,
                cwd=Path(__file__).parent
            )
            
            if result.returncode != 0:
                print(f" {demo_name} terminó con código de salida: {result.returncode}")
            else:
                print(f" {demo_name} completado exitosamente")
                
        except Exception as e:
            print(f" Error ejecutando {demo_name}: {e}")
        finally:
            print(f" {demo_name} finalizó")
        
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
        print(f"\n Señal {signum} recibida. Deteniendo todas las demos...")
        self.stop_all()

    def stop_all(self):
        """Detener todos los procesos"""
        self.running = False
        print(" Deteniendo procesos...")
        
        for i, process in enumerate(self.processes):
            if process.poll() is None:  # Si el proceso sigue ejecutándose
                try:
                    print(f" Deteniendo proceso {i+1}...")
                    process.terminate()
                    # Esperar 3 segundos para terminación suave
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        print(f" Forzando cierre del proceso {i+1}...")
                        process.kill()
                        process.wait()
                except Exception as e:
                    print(f" Error deteniendo proceso {i+1}: {e}")
        
        print(" Todos los procesos detenidos.")

    def run_triple_demo(self):
        """Ejecutar las tres demos simultáneamente"""
        print("""
╔═══════════════════════════════════════════════════════════════════╗
║                    TRIPLE DEMO EPSILON GREEDY                    ║
║          Comparación Simultánea de Tres Valores de Epsilon       ║
║                    CON IDENTIFICACIÓN VISUAL                     ║
╚═══════════════════════════════════════════════════════════════════╝

 Configuración Visual MEJORADA:
    Ventana 1: TITULO SÚPER LARGO "EPSILON 0.3 MODERADO" (Esquina Superior Izquierda)
      • Prefijo: [EPSILON-0.3-MODERADO] en todas las salidas
      • 30% exploración, 70% explotación
      • Comportamiento estratégico y dirigido
   
   Ventana 2: TITULO SÚPER LARGO "EPSILON 0.9 CAOTICO" (Centro Superior)
      • Prefijo: [EPSILON-0.9-CAOTICO] en todas las salidas
      • 90% exploración, 10% explotación
      • Comportamiento muy aleatorio y exploratorio
   
   Ventana 3: TITULO SÚPER LARGO "EPSILON VARIABLE INTERACTIVO" (Esquina Superior Derecha)
      • Prefijo: [EPSILON-VARIABLE-INTERACTIVO] en todas las salidas
      • Epsilon adaptativo según escenario
      • Puedes cambiar epsilon en tiempo real

 IDENTIFICACIÓN MÚLTIPLE Y ROBUSTA:
   1. TÍTULOS DE VENTANA: Súper largos con símbolos distintivos
   2. POSICIONES: Cada ventana en ubicación diferente de pantalla
   3. PREFIJOS CONSOLA: Cada salida tiene su identificador único
   4. METADATOS: Información distintiva en propiedades del stream

  Requisitos:
   • Ambiente Python con todas las dependencias instaladas
   • Archivos init.state y PokemonRed.gb en directorio padre
   • Suficiente memoria para ejecutar 3 emuladores simultáneamente

 Métricas:
   • Cada demo guardará sus métricas automáticamente
   • Usa Ctrl+C para detener todas las demos y guardar métricas
   • Revisa epsilon_greedy/results/ después de la ejecución

 Iniciando triple ejecución con identificación visual...
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
            print(" Iniciando Epsilon 0.3...")
            thread_03.start()
            time.sleep(2)
            
            print(" Iniciando Epsilon 0.9...")
            thread_09.start()
            time.sleep(2)
            
            print(" Iniciando Epsilon Interactivo...")
            thread_interactive.start()
            time.sleep(1)
            
            print("""
 Las tres demos han sido iniciadas correctamente.

IDENTIFICACIÓN VISUAL ACTIVA:
    Ventana "EPSILON 0.3 (MODERADO)" - Comportamiento balanceado
    Ventana "EPSILON 0.9 (CAÓTICO)" - Comportamiento muy exploratorio  
    Ventana "EPSILON VARIABLE (INTERACTIVO)" - Comportamiento adaptativo

 CONTROLES DISPONIBLES:
   • Ventana Verde (0.3): Automática (sin controles)
   • Ventana Roja (0.9): Automática (sin controles)
   • Ventana Azul (Variable): 
     - Teclas 1-7: Cambiar presets de epsilon
     - +/-: Ajustar epsilon manualmente
     - s: Ver estadísticas
     - q: Salir solo de esa ventana

DETENER TODO: Ctrl+C en esta terminal

OBSERVA LAS DIFERENCIAS EN LAS VENTANAS IDENTIFICADAS:
   Epsilon 0.3: Movimientos más consistentes y dirigidos
   Epsilon 0.9: Movimientos muy aleatorios y caóticos
   Epsilon Variable: Comportamiento que puedes modificar

TIP: Cambia el epsilon en la ventana azul (interactiva) para ver
   cómo se compara con las versiones fijas verde y roja!
""")
            
            # Esperar a que terminen todos los hilos
            for thread in self.threads:
                thread.join()
            
            if self.running:
                print("\n Triple ejecución completada normalmente.")
            
        except KeyboardInterrupt:
            print("\n Ctrl+C detectado. Deteniendo triple ejecución...")
            self.stop_all()
        except Exception as e:
            print(f"\n Error durante la ejecución: {e}")
            self.stop_all()
        
        print(f"""
 TRIPLE DEMO CON IDENTIFICACIÓN VISUAL FINALIZADA

 VENTANAS IDENTIFICADAS EJECUTADAS:
    "EPSILON 0.3 (MODERADO)" - Exploración balanceada
    "EPSILON 0.9 (CAÓTICO)" - Exploración muy alta
    "EPSILON VARIABLE (INTERACTIVO)" - Exploración adaptativa
 
 Revisa las métricas guardadas en:
    epsilon_greedy/results/
   
 Archivos generados por cada ventana identificada:
    demo_epsilon_03_metrics_[timestamp].md (Moderado)
    demo_epsilon_09_metrics_[timestamp].md (Caótico)
    epsilon_greedy_metrics_[timestamp].md (Variable - si se usó)
   
 Análisis sugerido:
   1. Compara las tasas de exploración vs explotación entre ventanas
   2. Observa diferencias en recompensas obtenidas por color de ventana
   3. Analiza qué epsilon fue más eficiente para obtener Pokemon
   4. Revisa los patrones de movimiento en los historiales de acciones

 Conclusiones esperadas:
    Epsilon 0.3: Balance óptimo entre exploración y explotación
    Epsilon 0.9: Exploración extensiva pero posiblemente ineficiente
    Epsilon Variable: Permite experimentar con diferentes estrategias
   
 NUEVA FUNCIONALIDAD: Identificación visual implementada exitosamente!
""")

def main():
    """Función principal"""
    runner = TripleEpsilonRunner()
    runner.run_triple_demo()

if __name__ == "__main__":
    main()