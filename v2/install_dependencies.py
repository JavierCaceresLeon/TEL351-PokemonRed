#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de instalación de dependencias agnóstico al sistema operativo.
Detecta automáticamente Windows/Linux y instala los paquetes correctos.

Uso:
    python install_dependencies.py
    python install_dependencies.py --gpu  # Para Linux con soporte CUDA
"""

import sys
import platform
import subprocess
import argparse

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def get_platform_info():
    """Detecta el sistema operativo y la arquitectura"""
    system = platform.system()
    machine = platform.machine()
    return {
        'system': system,
        'machine': machine,
        'is_windows': system == 'Windows',
        'is_linux': system == 'Linux',
        'is_mac': system == 'Darwin'
    }


def get_base_requirements():
    """Dependencias comunes a todos los sistemas operativos"""
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
        'numpy==2.1.2',
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


def get_pytorch_requirements(platform_info, gpu=False):
    """Dependencias de PyTorch según el sistema operativo y GPU"""
    if platform_info['is_windows']:
        # Windows: solo CPU (GPU requiere instalación manual)
        return ['torch==2.5.0']
    
    elif platform_info['is_linux']:
        if gpu:
            # Linux con GPU: PyTorch + CUDA 12.4
            return [
                'torch==2.5.0',
                'nvidia-cublas-cu12==12.4.5.8',
                'nvidia-cuda-cupti-cu12==12.4.127',
                'nvidia-cuda-nvrtc-cu12==12.4.127',
                'nvidia-cuda-runtime-cu12==12.4.127',
                'nvidia-cudnn-cu12==9.1.0.70',
                'nvidia-cufft-cu12==11.2.1.3',
                'nvidia-curand-cu12==10.3.5.147',
                'nvidia-cusolver-cu12==11.6.1.9',
                'nvidia-cusparse-cu12==12.3.1.170',
                'nvidia-nccl-cu12==2.21.5',
                'nvidia-nvjitlink-cu12==12.4.127',
                'nvidia-nvtx-cu12==12.4.127',
                'triton==3.1.0',
            ]
        else:
            # Linux sin GPU: solo CPU
            return ['torch==2.5.0']
    
    elif platform_info['is_mac']:
        # macOS: solo CPU
        return ['torch==2.5.0']
    
    return ['torch==2.5.0']


def install_packages(packages, description="paquetes"):
    """Instala una lista de paquetes usando pip"""
    print(f"\n Instalando {description}...")
    print(f"   Total: {len(packages)} paquetes")
    
    # Crear comando pip install
    cmd = [sys.executable, '-m', 'pip', 'install'] + packages
    
    try:
        subprocess.check_call(cmd)
        print(f" {description} instalados correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Error instalando {description}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Instalador de dependencias agnóstico al sistema operativo'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Instalar con soporte GPU (solo Linux con CUDA)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Mostrar qué se instalaría sin instalar'
    )
    
    args = parser.parse_args()
    
    # Validar versión de Python
    python_version = sys.version_info
    if python_version < (3, 10) or python_version >= (3, 13):
        print("\n" + "=" * 70)
        print("ERROR: Versión de Python incompatible")
        print("=" * 70)
        print(f"\nPython actual: {python_version.major}.{python_version.minor}.{python_version.micro}")
        print("\nPyBoy requiere Python 3.10, 3.11 o 3.12")
        print("   Python 3.13+ NO es compatible debido a cambios en Cython")
        print("\nSoluciones:")
        print("   1. Instalar Python 3.12: https://www.python.org/downloads/")
        print("   2. Usar pyenv (Linux/macOS): pyenv install 3.12")
        print("   3. Crear entorno conda: conda create -n pokeenv python=3.12")
        print("\nDespués de instalar Python 3.10-3.12, ejecuta:")
        print("   python3.12 install_dependencies.py")
        print("\n" + "=" * 70)
        sys.exit(1)
    
    # Detectar plataforma
    platform_info = get_platform_info()
    
    print("=" * 70)
    print(" Instalador de Dependencias - Pokemon Red RL Environment")
    print("=" * 70)
    print(f"\n  Sistema Operativo: {platform_info['system']}")
    print(f"  Arquitectura: {platform_info['machine']}")
    print(f" Python: {sys.version.split()[0]}")
    
    # Validar GPU en Windows
    if args.gpu and platform_info['is_windows']:
        print("\n  ADVERTENCIA: Soporte GPU en Windows requiere instalación manual.")
        print("   Se instalará versión CPU. Para GPU, sigue las instrucciones en:")
        print("   https://pytorch.org/get-started/locally/")
        args.gpu = False
    
    # Obtener dependencias
    base_reqs = get_base_requirements()
    pytorch_reqs = get_pytorch_requirements(platform_info, args.gpu)
    all_reqs = base_reqs + pytorch_reqs
    
    # Modo dry-run
    if args.dry_run:
        print(f"\n Paquetes a instalar ({len(all_reqs)} total):")
        print("\n--- Dependencias base ---")
        for pkg in base_reqs:
            print(f"  • {pkg}")
        print("\n--- PyTorch y dependencias ---")
        for pkg in pytorch_reqs:
            print(f"  • {pkg}")
        return
    
    # Instalación
    print(f"\n Modo de instalación:")
    if platform_info['is_windows']:
        print("Windows: PyTorch CPU")
    elif platform_info['is_linux']:
        if args.gpu:
            print("Linux: PyTorch + CUDA 12.4 (GPU)")
        else:
            print("Linux: PyTorch CPU")
    else:
        print(f"{platform_info['system']}: PyTorch CPU")
    
    # Confirmar instalación
    response = input("\n¿Continuar con la instalación? [S/n]: ").strip().lower()
    if response and response not in ['s', 'y', 'yes', 'si', 'sí']:
        print(" Instalación cancelada")
        return
    
    # Instalar dependencias base
    success = install_packages(base_reqs, "dependencias base")
    if not success:
        print("\n Fallo en instalación de dependencias base")
        sys.exit(1)
    
    # Instalar PyTorch y dependencias
    success = install_packages(pytorch_reqs, "PyTorch y dependencias")
    if not success:
        print("\n  Advertencia: Algunas dependencias de PyTorch fallaron")
        print("   El entorno puede funcionar parcialmente")
    
    # Verificar instalación
    print("\n" + "=" * 70)
    print(" Verificando instalación...")
    print("=" * 70)
    
    try:
        import torch
        print(f"\n PyTorch {torch.__version__} instalado")
        print(f"   CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\nPyTorch no se pudo importar")
        sys.exit(1)
    
    try:
        import gymnasium
        print(f" Gymnasium {gymnasium.__version__} instalado")
    except ImportError:
        print(" Gymnasium no se pudo importar")
        sys.exit(1)
    
    try:
        import pyboy
        print(f" PyBoy instalado")
    except ImportError:
        print(" PyBoy no se pudo importar")
        sys.exit(1)
    
    try:
        import stable_baselines3
        print(f" Stable-Baselines3 {stable_baselines3.__version__} instalado")
    except ImportError:
        print(" Stable-Baselines3 no se pudo importar")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print(" ¡Instalación completada exitosamente!")
    print("=" * 70)
    print("\n Próximos pasos:")
    print("   1. Ejecutar: python run_pretrained_interactive.py")
    print("   2. O: python baseline_fast_v2.py")
    print("\n Para más información, consulta el README.md")


if __name__ == '__main__':
    main()
