"""
Script de instalación automática para el proyecto de comparación
Instala todas las dependencias necesarias
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Instalar un paquete usando pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package_name):
    """Verificar si un paquete está instalado"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    """Instalar todas las dependencias"""
    print("=== Instalador de Dependencias ===")
    print("Instalando dependencias para la comparación de agentes...")
    
    # Lista de paquetes requeridos
    packages = [
        ("numpy", "numpy"),
        ("scikit-image", "skimage"),
        ("matplotlib", "matplotlib"),
        ("pyboy", "pyboy"),
        ("stable-baselines3", "stable_baselines3"),
        ("torch", "torch"),
        ("mediapy", "mediapy"),
        ("einops", "einops"),
        ("pandas", "pandas"),
        ("seaborn", "seaborn")
    ]
    
    print(f"\nInstalando {len(packages)} paquetes...")
    
    success_count = 0
    
    for package_pip, package_import in packages:
        print(f"\n--- {package_pip} ---")
        
        # Verificar si ya está instalado
        if check_package(package_import):
            print(f"✓ {package_pip} ya está instalado")
            success_count += 1
            continue
        
        # Intentar instalar
        print(f"Instalando {package_pip}...")
        if install_package(package_pip):
            print(f"✓ {package_pip} instalado correctamente")
            success_count += 1
        else:
            print(f"✗ Error instalando {package_pip}")
    
    print(f"\n=== Resumen ===")
    print(f"Paquetes instalados exitosamente: {success_count}/{len(packages)}")
    
    if success_count == len(packages):
        print("🎉 ¡Todas las dependencias instaladas!")
        print("\nAhora puedes ejecutar:")
        print("  python test_setup.py")
        print("  python run_comparison.py")
    else:
        print("⚠️ Algunas dependencias no se pudieron instalar.")
        print("Instala manualmente con:")
        print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main()
