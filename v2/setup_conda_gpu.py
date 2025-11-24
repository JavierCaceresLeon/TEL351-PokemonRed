#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de configuración de entorno Conda 'pokeenv' con soporte GPU y Numba.
Diseñado para Windows/Linux para maximizar rendimiento mediante el uso de GPU y compilación JIT.

Este script:
1. Crea un entorno Conda llamado 'pokeenv'.
2. Instala PyTorch con soporte CUDA (GPU).
3. Instala Numba y cudatoolkit para aceleración.
4. Instala el resto de dependencias del proyecto.

Uso:
    python setup_conda_gpu.py
"""

import sys
import subprocess
import shutil
import platform

def check_conda():
    """Verifica si conda está instalado y disponible en el PATH"""
    if shutil.which("conda") is None:
        print("Error: Conda no encontrado. Por favor instala Anaconda o Miniconda y asegúrate de que esté en el PATH.")
        print("Descarga: https://www.anaconda.com/download o https://docs.conda.io/en/latest/miniconda.html")
        sys.exit(1)
    print("✓ Conda detectado correctamente.")

def create_conda_env(env_name="pokeenv", python_version="3.11"):
    """Crea el entorno conda si no existe"""
    print(f"\n[1/4] Creando/Verificando entorno '{env_name}' con Python {python_version}...")
    
    # Verificar si el entorno ya existe
    try:
        envs = subprocess.check_output(["conda", "env", "list"]).decode("utf-8")
        if env_name in envs:
            print(f"  El entorno '{env_name}' ya existe. Se procederá a actualizar/instalar paquetes.")
            return
    except subprocess.CalledProcessError:
        pass

    try:
        subprocess.check_call(["conda", "create", "-n", env_name, f"python={python_version}", "-y"])
        print(f"  Entorno '{env_name}' creado exitosamente.")
    except subprocess.CalledProcessError as e:
        print(f"  Error creando el entorno: {e}")
        sys.exit(1)

def get_pip_requirements():
    """Retorna la lista de paquetes a instalar vía pip (excluyendo los de conda)"""
    return [
        'absl-py==2.1.0',
        'asttokens==2.4.1',
        'cloudpickle==3.1.0',
        'contourpy==1.3.0',
        'cycler==0.12.1',
        'decorator==5.1.1',
        'einops==0.8.0',
        'executing==2.1.0',
        'Farama-Notifications==0.0.4',
        'filelock==3.16.1',
        'fonttools==4.54.1',
        'fsspec==2024.9.0',
        'grpcio==1.67.0',
        'gymnasium==0.29.1',
        'imageio==2.36.0',
        'ipython==8.28.0',
        'jedi==0.19.1',
        'Jinja2==3.1.4',
        'kiwisolver==1.4.7',
        'lazy_loader==0.4',
        'Markdown==3.7',
        'MarkupSafe==3.0.2',
        'matplotlib==3.9.2',
        'matplotlib-inline==0.1.7',
        'mediapy==1.2.2',
        'mpmath==1.3.0',
        'networkx==3.4.1',
        # 'numpy' -> Instalado por Conda
        'packaging==24.1',
        'pandas==2.2.3',
        'parso==0.8.4',
        'pillow==11.0.0',
        'prompt_toolkit==3.0.48',
        'protobuf==5.28.2',
        'pure_eval==0.2.3',
        'pyboy==2.4.0',
        'Pygments==2.18.0',
        'pyparsing==3.2.0',
        'PySDL2==0.9.16',
        'pysdl2-dll==2.30.2',
        'python-dateutil==2.9.0.post0',
        'pytz==2024.2',
        'scikit-image==0.24.0',
        'scipy==1.14.1',
        'setuptools==75.1.0',
        'six==1.16.0',
        'stable_baselines3==2.3.2',
        'stack-data==0.6.3',
        'sympy==1.13.1',
        'tensorboard==2.18.0',
        'tensorboard-data-server==0.7.2',
        'tifffile==2024.9.20',
        'traitlets==5.14.3',
        'typing_extensions==4.12.2',
        'tzdata==2024.2',
        'wcwidth==0.2.13',
        'websockets==13.1',
        'Werkzeug==3.0.4',
        'wheel==0.44.0',
        'psutil==7.0.0',
        'seaborn==0.13.2',
        'scikit-learn==1.7.2',
    ]

def install_conda_packages(env_name):
    """Instala paquetes optimizados vía Conda (PyTorch GPU, Numba, NumPy)"""
    print("\n[2/4] Instalando PyTorch (GPU), Numba y NumPy vía Conda...")
    print("      Esto puede tardar unos minutos dependiendo de tu conexión...")
    
    # Paquetes clave para rendimiento
    # pytorch-cuda=12.1 es una versión estable y compatible con la mayoría de GPUs modernas
    packages = [
        "pytorch", 
        "torchvision", 
        "torchaudio", 
        "pytorch-cuda=12.1", 
        "numba", 
        "numpy",
        "cudatoolkit"
    ]
    
    channels = ["pytorch", "nvidia", "conda-forge"]
    
    cmd = ["conda", "install", "-n", env_name, "-y"]
    for ch in channels:
        cmd.extend(["-c", ch])
    cmd.extend(packages)
    
    try:
        subprocess.check_call(cmd)
        print("✓ Paquetes Conda instalados correctamente.")
    except subprocess.CalledProcessError as e:
        print(f"Error instalando paquetes Conda: {e}")
        print("Intentando instalación sin especificar versión de CUDA (fallback)...")
        # Fallback simple
        cmd_fallback = ["conda", "install", "-n", env_name, "-y", "pytorch", "torchvision", "torchaudio", "cpuonly", "-c", "pytorch"]
        if platform.system() == "Linux" or platform.system() == "Windows":
             cmd_fallback = ["conda", "install", "-n", env_name, "-y", "pytorch", "torchvision", "torchaudio", "pytorch-cuda=11.8", "-c", "pytorch", "-c", "nvidia"]
        
        try:
            subprocess.check_call(cmd_fallback)
        except:
            print("Fallo crítico en instalación de Conda. Revisa tu conexión o configuración de Conda.")
            sys.exit(1)

def install_pip_packages(env_name, packages):
    """Instala el resto de paquetes vía pip dentro del entorno conda"""
    print(f"\n[3/4] Instalando {len(packages)} paquetes adicionales vía pip...")
    
    # Usamos 'conda run' para asegurar que usamos el pip del entorno
    # En Windows, a veces es mejor llamar al pip ejecutable directamente si conda run falla, 
    # pero conda run es más estándar.
    
    cmd = ["conda", "run", "-n", env_name, "pip", "install"] + packages
    
    try:
        subprocess.check_call(cmd)
        print("✓ Paquetes pip instalados correctamente.")
    except subprocess.CalledProcessError as e:
        print(f"Error instalando paquetes pip: {e}")
        sys.exit(1)

def verify_installation(env_name):
    """Verifica que los componentes críticos funcionen"""
    print("\n[4/4] Verificando instalación...")
    
    verification_script = """
import torch
import numba
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("ADVERTENCIA: CUDA no detectado. PyTorch usará CPU.")

print(f"Numba: {numba.__version__}")
try:
    from numba import cuda
    print(f"Numba CUDA disponible: {cuda.is_available()}")
except:
    print("Numba CUDA check falló")
"""
    
    cmd = ["conda", "run", "-n", env_name, "python", "-c", verification_script]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print("Error en la verificación.")

def main():
    print("="*70)
    print(" Configuración de Entorno 'pokeenv' (GPU + Numba)")
    print("="*70)
    
    check_conda()
    
    create_conda_env()
    
    install_conda_packages("pokeenv")
    
    pip_reqs = get_pip_requirements()
    install_pip_packages("pokeenv", pip_reqs)
    
    verify_installation("pokeenv")
    
    print("\n" + "="*70)
    print(" ¡Instalación completada exitosamente!")
    print("="*70)
    print(" Para empezar a usar el entorno:")
    print("     conda activate pokeenv")
    print("     python run_pretrained_interactive.py")
    print("="*70)

if __name__ == "__main__":
    main()
