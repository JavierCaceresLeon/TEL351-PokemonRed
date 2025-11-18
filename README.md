[Video TAREA 1 DEMOSTRACIÓN](https://youtu.be/EyTkha_VWgY)
# Entrenamiento de Agentes de Aprendizaje por Refuerzo para Pokémon Red

Este proyecto implementa un entorno de aprendizaje por refuerzo para entrenar agentes de IA que jueguen Pokémon Red automáticamente. El agente aprende a navegar por el mundo del juego, capturar Pokémon, luchar en batallas y completar objetivos usando técnicas de aprendizaje profundo.

## Descripción General del Proyecto

El proyecto utiliza PyBoy (un emulador de Game Boy) junto con Stable Baselines3 para crear un entorno de gimnasio donde los agentes pueden interactuar con Pokémon Red. El agente observa las pantallas del juego y aprende políticas óptimas mediante algoritmos como PPO (Proximal Policy Optimization).

## Instalación y Configuración (V2 - Recomendada)

> **IMPORTANTE**: Esta guía está diseñada para ejecutar la **versión V2** del proyecto, que es la versión mejorada y recomendada. V2 incluye optimizaciones significativas y es mucho más fácil de instalar gracias a su compatibilidad cross-platform automática.

> **PyBoy NO funciona con Python 3.13. **

Esta instalación es **agnóstica al sistema operativo**.

### Requisitos Previos

- **Python 3.10, 3.11 o 3.12** **Python 3.13+ NO es compatible con PyBoy**
- **Python 3.10, 3.11 o 3.12** **Python 3.13+ NO es compatible con PyBoy**
- **pip 21.0+** (para soporte de marcadores de entorno)
- **ROM de Pokémon Red** legalmente obtenida (1MB, sha1: `ea9bcae617fdf159b045185467ae58b2e4a48b9a`)

---

### Instalación Rápida - Script Automático

```bash
# 1. Clonar el repositorio
git clone https://github.com/JavierCaceresLeon/TEL351-PokemonRed.git
cd TEL351-PokemonRed

# 2. Crear entorno conda con Python 3.10 (recomendado)
conda create -n pokeenv python=3.10.19

# Activar entorno:
conda activate pokeenv
# 2. Crear entorno conda con Python 3.10 (recomendado)
conda create -n pokeenv python=3.10.19

# Activar entorno:
conda activate pokeenv

# 3. Navegar a v2 y ejecutar instalador
cd v2
python install_dependencies.py

# 4. Verificar instalación
python -c "import torch; import pyboy; import gymnasium; print('Instalación exitosa')"
python -c "import torch; import pyboy; import gymnasium; print('Instalación exitosa')"

# 5. Ejecutar modelo preentrenado
python run_pretrained_interactive.py
```

**Opciones del script:**
```bash
python install_dependencies.py              # Instalación CPU (todas las plataformas)
python install_dependencies.py --gpu        # Linux con soporte CUDA GPU
python install_dependencies.py --dry-run    # Ver qué se instalará sin instalar
```

---

### Compatibilidad Automática por Sistema Operativo (no testeado)

El archivo `v2/requirements.txt` usa **marcadores de entorno de pip** para instalar solo paquetes compatibles:

```python
# Ejemplo de marcadores:
nvidia-nccl-cu12==2.21.5; sys_platform == 'linux'  # Solo Linux
triton==3.1.0; sys_platform == 'linux'              # Solo Linux
torch==2.5.0                                         # Todos los OS
```

**Resultado por plataforma:**
- **Windows:** 60 paquetes (omite automáticamente 13 paquetes Linux-only)
- **Linux:** 73 paquetes (incluye NVIDIA CUDA + triton)
- **macOS:** 60 paquetes (omite automáticamente paquetes incompatibles)

---

### Verificación de Instalación

Verifica que todo funcione correctamente:

```python
python -c "
import torch
import pyboy
import gymnasium
import stable_baselines3
import websockets

print('PyTorch:', torch.__version__)
print('CUDA disponible:', torch.cuda.is_available())
print('PyBoy instalado')
print('Gymnasium:', gymnasium.__version__)
print('Stable-Baselines3:', stable_baselines3.__version__)
print('\n¡Todas las dependencias instaladas correctamente!')
"
```

---

### Solución de Problemas Comunes

<details>
<summary><b>Error: "No matching distribution found for nvidia-nccl-cu12"</b></summary>

**Causa:** Estás usando un `requirements.txt` antiguo.

**Solución:**
```bash
cd v2
git pull origin main  # Actualizar
python install_dependencies.py  # O usar script automático
```

</details>

<details>
<summary><b>Error: "ModuleNotFoundError: No module named 'websockets'"</b></summary>

**Solución:**
```bash
pip install websockets==13.1
```

</details>

<details>
<summary><b>Error: PyBoy API incompatibilidad</b></summary>

Los wrappers ya están implementados. Si persiste:

```bash
pip install pyboy==2.4.0 --force-reinstall
```

</details>

<details>
<summary><b>Error: PyBoy compilation error (Cython) en Python 3.13</b></summary>

**Causa:** Python 3.13 no es compatible con PyBoy 2.4.0.

**Error típico:**
```
Cython.Compiler.Errors.CompileError: pyboy\core\cartridge\cartridge.py
Unicode objects only support coercion to Py_UNICODE*
```

**Solución:** Instalar Python 3.10, 3.11 o 3.12:

**Windows:**
```bash
# Descargar Python 3.12 desde python.org
# O usar chocolatey:
choco install python --version=3.12.0
```

**Linux:**
```bash
# Opción 1: pyenv
pyenv install 3.12.0
pyenv local 3.12.0

# Opción 2: deadsnakes PPA (Ubuntu)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv
```

**Conda (todas las plataformas):**
```bash
conda create -n pokeenv python=3.12
conda activate pokeenv
```

</details>

<details>
<summary><b>Error: PyBoy compilation error (Cython) en Python 3.13</b></summary>

**Causa:** Python 3.13 no es compatible con PyBoy 2.4.0.

**Error típico:**
```
Cython.Compiler.Errors.CompileError: pyboy\core\cartridge\cartridge.py
Unicode objects only support coercion to Py_UNICODE*
```

**Solución:** Instalar Python 3.10, 3.11 o 3.12:

**Windows:**
```bash
# Descargar Python 3.12 desde python.org
# O usar chocolatey:
choco install python --version=3.12.0
```

**Linux:**
```bash
# Opción 1: pyenv
pyenv install 3.12.0
pyenv local 3.12.0

# Opción 2: deadsnakes PPA (Ubuntu)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv
```

**Conda (todas las plataformas):**
```bash
conda create -n pokeenv python=3.12
conda activate pokeenv
```

</details>

<details>
<summary><b>Windows: SDL2 DLL no encontrado</b></summary>

**Solución:**
```bash
pip install pysdl2-dll==2.30.2 --force-reinstall
```

</details>

---

### Por Qué V2 es la Versión Recomendada

**Mejoras sobre baseline:**
-  **3x más rápido** en entrenamiento
-  **Mejor exploración** (basada en coordenadas vs KNN)
-  **Menos bloqueos** (manejo mejorado de menús)
-  **Alcanza Cerulean City** consistentemente
-  **Instalación simplificada** (agnóstica al OS)
-  **Streaming integrado** (visualización del mapa en tiempo real)

---
