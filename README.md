[Video TAREA 1 DEMOSTRACIÓN](https://youtu.be/EyTkha_VWgY)
# Entrenamiento de Agentes de Aprendizaje por Refuerzo para Pokémon Red

Este proyecto implementa un entorno de aprendizaje por refuerzo para entrenar agentes de IA que jueguen Pokémon Red automáticamente. El agente aprende a navegar por el mundo del juego, capturar Pokémon, luchar en batallas y completar objetivos usando técnicas de aprendizaje profundo.

## Descripción General del Proyecto

El proyecto utiliza PyBoy (un emulador de Game Boy) junto con Stable Baselines3 para crear un entorno de gimnasio donde los agentes pueden interactuar con Pokémon Red. El agente observa las pantallas del juego y aprende políticas óptimas mediante algoritmos como PPO (Proximal Policy Optimization).

## Instalación y Configuración (V2 - Recomendada)

> **IMPORTANTE**: Esta guía está diseñada para ejecutar la **versión V2** del proyecto, que es la versión mejorada y recomendada. V2 incluye optimizaciones significativas y es mucho más fácil de instalar gracias a su compatibilidad cross-platform automática.

Esta instalación es **agnóstica al sistema operativo** y funciona en Windows, Linux y macOS sin modificaciones manuales.

### Requisitos Previos

- **Python 3.10+** (recomendado para máxima compatibilidad)
- **pip 21.0+** (para soporte de marcadores de entorno)
- **ROM de Pokémon Red** legalmente obtenida (1MB, sha1: `ea9bcae617fdf159b045185467ae58b2e4a48b9a`)
- **Sistema Operativo:** Windows 10+, Linux (Ubuntu 20.04+), o macOS 11+
- **RAM:** Mínimo 4GB (recomendado 8GB para entrenamiento)

---

### Instalación Rápida - Opción 1: Script Automático (Más Fácil)

El script detecta automáticamente tu sistema operativo e instala las dependencias correctas:

```bash
# 1. Clonar el repositorio
git clone https://github.com/JavierCaceresLeon/TEL351-PokemonRed.git
cd TEL351-PokemonRed

# 2. Copiar ROM de Pokémon Red al directorio raíz
# Renombrar como PokemonRed.gb exactamente

# 3. Navegar a v2 y ejecutar instalador
cd v2
python install_dependencies.py

# 4. Verificar instalación
python -c "import torch; import pyboy; import gymnasium; print('✅ Instalación exitosa')"

# 5. Ejecutar modelo preentrenado
python run_pretrained_interactive.py
```

**Ventajas del script automático:**
- Detecta Windows/Linux/macOS automáticamente
- Instala solo dependencias compatibles con tu sistema
- Valida la instalación automáticamente
- Soporte para GPU en Linux con `--gpu`

**Opciones del script:**
```bash
python install_dependencies.py              # Instalación CPU (todas las plataformas)
python install_dependencies.py --gpu        # Linux con soporte CUDA GPU
python install_dependencies.py --dry-run    # Ver qué se instalará sin instalar
```

---

### Instalación Rápida - Opción 2: Manual con pip

Si prefieres control manual, usa directamente `requirements.txt` que es **agnóstico al sistema operativo**:

```bash
# 1. Clonar repositorio
git clone https://github.com/JavierCaceresLeon/TEL351-PokemonRed.git
cd TEL351-PokemonRed

# 2. Crear entorno virtual (recomendado)
python -m venv venv

# Activar entorno:
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Actualizar pip
python -m pip install --upgrade pip

# 4. Copiar ROM de Pokémon Red al directorio raíz
# Renombrar como PokemonRed.gb

# 5. Instalar dependencias V2
cd v2
pip install -r requirements.txt
# El archivo usa marcadores de entorno para compatibilidad automática

# 6. Verificar instalación
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 7. Ejecutar modelo preentrenado
python run_pretrained_interactive.py
```

---

### Compatibilidad Automática por Sistema Operativo

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
<summary><b>Windows: SDL2 DLL no encontrado</b></summary>

**Solución:**
```bash
pip install pysdl2-dll==2.30.2 --force-reinstall
```

</details>

---

### Recursos Adicionales de Instalación

- **Guía Detallada:** `v2/INSTALLATION.md` (instrucciones completas por OS)
- **Troubleshooting Completo:** `CATASTRO_ERRORES_CORREGIDOS.md`
- **Solución Cross-Platform:** `v2/CROSS_PLATFORM_SOLUTION.md`
- **PyTorch con GPU:** https://pytorch.org/get-started/locally/

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
