# Setup Instructions for Pokemon Red Agent Comparison

## Environment Setup

Este proyecto requiere un entorno Python específico debido a las dependencias de CUDA y PyBoy. Sigue estas instrucciones para configurar correctamente el entorno.

### Opción 1: Conda (Recomendado)

```bash
# 1. Crear entorno desde el archivo YAML
cd comparison_agents
conda env create -f environment.yml

# 2. Activar el entorno
conda activate pokemon-red-comparison

# 3. Verificar instalación
python --version  # Debería mostrar 3.10.18
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pyboy; print('PyBoy instalado correctamente')"
python -c "import stable_baselines3; print('SB3 instalado correctamente')"
```

### Opción 2: Conda Manual

```bash
# 1. Crear entorno con Python 3.10
conda create -n pokemon-red-comparison python=3.10.18

# 2. Activar entorno
conda activate pokemon-red-comparison

# 3. Instalar PyTorch (compatible con Python 3.10)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. Instalar dependencias científicas
conda install numpy pandas matplotlib scipy scikit-learn scikit-image seaborn plotly jupyter ipython -c conda-forge

# 5. Instalar dependencias específicas con pip
pip install -r requirements_py310.txt
```

### Opción 3: venv (Alternativa)

```bash
# 1. Crear entorno virtual (asegúrate de tener Python 3.10 disponible)
python3.10 -m venv pokemon_red_env

# 2. Activar entorno
# Windows:
pokemon_red_env\Scripts\activate
# Linux/Mac:
source pokemon_red_env/bin/activate

# 3. Actualizar pip
python -m pip install --upgrade pip

# 4. Instalar dependencias
pip install -r requirements_py310.txt
```

## Verificación del Entorno

Después de la instalación, ejecuta este script para verificar que todo esté funcionando:

```bash
cd comparison_agents
python example_usage.py
```

O ejecuta estos comandos para verificación manual:

```python
# Verificar dependencias principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import stable_baselines3
import pyboy
import gymnasium
import mediapy
import einops

print("✓ Todas las dependencias principales instaladas correctamente")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
```

## Solución de Problemas Comunes

### Error: nvidia-nccl-cu12 no encontrado
- **Causa**: Versiones de CUDA que requieren Python 3.11+
- **Solución**: Usar Python 3.10 con CUDA 11.8 (incluido en nuestro setup)

### Error: PyBoy no se instala
```bash
# Instalar dependencias del sistema (Ubuntu/Debian)
sudo apt-get install libsdl2-dev

# Windows: Las DLLs se instalan automáticamente con pysdl2-dll
pip install pysdl2-dll
```

### Error: PyTorch sin CUDA
```bash
# Reinstalar PyTorch con soporte CUDA
pip uninstall torch
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Problemas de memoria con gymnasium
```bash
# Instalar versión específica compatible
pip install gymnasium==0.29.1
```

## Configuración Específica para el Proyecto

Una vez instalado el entorno, configura las rutas del proyecto:

```bash
# Verificar que los archivos del juego estén disponibles
ls ../PokemonRed.gb
ls ../init.state

# Crear directorios de salida
mkdir -p comparison_results
mkdir -p metrics_analysis
mkdir -p experiments
```

## Execution Commands

```bash
# Activar entorno
conda activate pokemon-red-comparison

# Ejecutar comparación básica
python run_comparison.py --mode standalone --episodes 3

# Ejecutar comparación completa
python run_comparison.py --mode full --episodes 5

# Ejecutar ejemplos
python example_usage.py
```

## Notas Importantes

1. **GPU**: Si tienes GPU NVIDIA, el entorno debería detectarla automáticamente
2. **Memoria**: Los experimentos pueden usar mucha RAM, considera reducir `--episodes` si es necesario
3. **Tiempo**: Cada episodio puede tomar 5-15 minutos dependiendo de `max_steps`
4. **Archivos**: Asegúrate de que `PokemonRed.gb` e `init.state` estén en el directorio padre

## Desactivar Entorno

```bash
conda deactivate
```

## Eliminar Entorno (si es necesario)

```bash
conda env remove -n pokemon-red-comparison
```
