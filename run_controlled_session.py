#!/usr/bin/env python3
"""
Script para ejecutar sesiones de análisis controladas
Útil para generar datos sin intervención manual
"""

import subprocess
import time
import signal
import os
import sys
from pathlib import Path
import psutil

class SessionController:
    def __init__(self, session_duration_minutes=10):
        self.duration = session_duration_minutes * 60
        self.process = None
        self.session_dir = None
        
    def run_controlled_session(self, headless=True):
        """Ejecuta una sesión controlada del agente"""
        print(f"🚀 Iniciando sesión de {self.duration/60:.1f} minutos...")
        
        # Configurar comando
        if headless:
            # Modificar temporalmente el archivo para modo headless
            self._modify_config_for_headless()
        
        try:
            # Iniciar proceso
            self.process = subprocess.Popen(
                ['python', 'run_pretrained_interactive.py'],
                cwd='v2',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            print(f"📊 Proceso iniciado (PID: {self.process.pid})")
            print(f"⏱️ Esperando {self.duration/60:.1f} minutos...")
            
            # Esperar duración especificada
            time.sleep(self.duration)
            
            # Terminar proceso limpiamente
            self._terminate_gracefully()
            
            # Buscar directorio de sesión generado
            self.session_dir = self._find_latest_session()
            
            if self.session_dir:
                print(f"✅ Sesión completada: {self.session_dir}")
                return self.session_dir
            else:
                print("⚠️ No se encontró directorio de sesión")
                return None
                
        except KeyboardInterrupt:
            print("\n🛑 Sesión interrumpida por usuario")
            self._terminate_gracefully()
            return None
        except Exception as e:
            print(f"❌ Error durante la sesión: {e}")
            return None
        finally:
            self._restore_config()
    
    def _modify_config_for_headless(self):
        """Modifica temporalmente la configuración para modo headless"""
        config_file = Path('v2/run_pretrained_interactive.py')
        if config_file.exists():
            # Crear backup
            backup_file = config_file.with_suffix('.py.backup')
            config_file.replace(backup_file)
            
            # Leer archivo original
            with open(backup_file, 'r') as f:
                content = f.read()
            
            # Modificar para headless
            modified_content = content.replace(
                "'headless': False,",
                "'headless': True,"
            )
            
            # Escribir archivo modificado
            with open(config_file, 'w') as f:
                f.write(modified_content)
    
    def _restore_config(self):
        """Restaura la configuración original"""
        config_file = Path('v2/run_pretrained_interactive.py')
        backup_file = config_file.with_suffix('.py.backup')
        
        if backup_file.exists():
            backup_file.replace(config_file)
    
    def _terminate_gracefully(self):
        """Termina el proceso de forma limpia"""
        if self.process and self.process.poll() is None:
            try:
                # Intentar terminación suave
                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)
                
                # Terminar hijos primero
                for child in children:
                    child.terminate()
                
                # Terminar proceso principal
                parent.terminate()
                
                # Esperar un poco
                time.sleep(2)
                
                # Forzar si es necesario
                if parent.is_running():
                    parent.kill()
                    
                print("🔄 Proceso terminado limpiamente")
                
            except Exception as e:
                print(f"⚠️ Error terminando proceso: {e}")
    
    def _find_latest_session(self):
        """Encuentra el directorio de sesión más reciente"""
        session_dirs = list(Path('v2').glob('session_*'))
        if session_dirs:
            # Ordenar por tiempo de modificación
            latest = max(session_dirs, key=lambda p: p.stat().st_mtime)
            return latest.name
        return None
    
    def analyze_session(self, session_name=None):
        """Analiza la sesión generada"""
        if session_name is None:
            session_name = self.session_dir
        
        if session_name:
            print(f"\n🔍 Analizando sesión: {session_name}")
            subprocess.run([
                'python', 'analyze_session.py', f'v2/{session_name}'
            ])
        else:
            print("❌ No hay sesión para analizar")

def main():
    if len(sys.argv) < 2:
        print("Uso: python run_controlled_session.py <minutos> [headless]")
        print("\nEjemplos:")
        print("  python run_controlled_session.py 5        # 5 minutos con ventana")
        print("  python run_controlled_session.py 10 True  # 10 minutos sin ventana")
        return
    
    try:
        duration = float(sys.argv[1])
        headless = len(sys.argv) > 2 and sys.argv[2].lower() == 'true'
        
        controller = SessionController(duration)
        session_dir = controller.run_controlled_session(headless)
        
        if session_dir:
            # Analizar automáticamente
            controller.analyze_session(session_dir)
            
            print(f"\n📋 Resumen:")
            print(f"  • Duración: {duration} minutos")
            print(f"  • Modo: {'Headless' if headless else 'Con ventana'}")
            print(f"  • Sesión: {session_dir}")
            print(f"  • Análisis: Completado")
        
    except ValueError:
        print("❌ Error: La duración debe ser un número")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    main()
