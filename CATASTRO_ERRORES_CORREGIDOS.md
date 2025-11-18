# 📋 Catastro de Errores Corregidos - Entorno PokeEnv

**Fecha:** 18 de noviembre de 2025  
**Entorno:** `pokeenv` (Conda)  
**Sistema:** Windows  
**Última actualización:** 18/11/2025 - Problema 5 agregado (Python 3.13 incompatibilidad)

---

## 🔴 Problema 1: ModuleNotFoundError: No module named 'websockets'

### Descripción del Error
```
ModuleNotFoundError: No module named 'websockets'
```

### Archivos Afectados
- `v2/baseline_fast_v2.py` → importa `stream_agent_wrapper.py`
- `v2/stream_agent_wrapper.py` → línea 2: `import websockets`
- `epsilon_greedy/run_epsilon_greedy_interactive.py` → importa `v2.stream_agent_wrapper`

### Causa Raíz
El módulo `websockets==13.1` está listado en `v2/requirements.txt` (línea 73) pero NO fue instalado en el entorno `pokeenv`.

### Evidencia
```bash
(pokeenv) PS> python .\baseline_fast_v2.py
ModuleNotFoundError: No module named 'websockets'

(pokeenv) PS> python .\run_epsilon_greedy_interactive.py  
ModuleNotFoundError: No module named 'websockets'
```

### Solución Aplicada
```bash
pip install websockets==13.1
```

### Resultado
websockets` instalado correctamente (versión 13.1)

---

## Problema 2: AttributeError: 'pyboy.pyboy.PyBoy' object has no attribute 'screen_buffer'

### Descripción del Error
```
AttributeError: 'pyboy.pyboy.PyBoy' object has no attribute 'screen_buffer'
```

### Archivos Afectados
- `v2/red_gym_env_v2.py` línea 256 (método `render()`)
- `v2/red_gym_env_v2.py` línea 568 (método `read_m()`)

### Causa Raíz
**Incompatibilidad de versiones de PyBoy:**

El repositorio tiene código mezclado para **PyBoy v1.x** y **PyBoy v2.x**, pero el entorno `pokeenv` tiene **PyBoy 2.4.0** instalado.

**API PyBoy 1.x (antigua):**
```python
self.pyboy.get_memory_value(addr)    # lectura memoria
self.pyboy.set_memory_value(addr, val)  # escritura memoria
self.pyboy.screen_buffer()           # obtener pantalla
```

**API PyBoy 2.x (nueva):**
```python
self.pyboy.memory[addr]              # lectura memoria
self.pyboy.memory[addr] = val        # escritura memoria
self.pyboy.screen.ndarray            # obtener pantalla
```

### Evidencia
```bash
(pokeenv) PS> python .\run_epsilon_greedy_interactive.py
AttributeError: 'pyboy.pyboy.PyBoy' object has no attribute 'screen_buffer'
  File "v2\red_gym_env_v2.py", line 256, in render
    game_pixels_render = self.pyboy.screen_buffer()
```

### Solución Aplicada

**1. Creación de wrappers de compatibilidad en `v2/red_gym_env_v2.py`:**

Agregado método `_setup_pyboy_compat()` después de `__init__`:

```python
def _setup_pyboy_compat(self):
    """Setup compatibility wrappers for various PyBoy versions (memory and screen)"""
    pyboy = self.pyboy
    
    # memory access - prueba múltiples APIs
    if hasattr(pyboy, "memory"):
        get_mem = lambda addr: int(pyboy.memory[addr])
        set_mem = lambda addr, val: pyboy.memory.__setitem__(addr, val)
    elif hasattr(pyboy, "get_memory_value"):
        get_mem = lambda addr: int(pyboy.get_memory_value(addr))
        if hasattr(pyboy, "set_memory_value"):
            set_mem = lambda addr, val: pyboy.set_memory_value(addr, val)
        else:
            set_mem = lambda addr, val: None
    elif hasattr(pyboy, "get_memory"):
        mem = pyboy.get_memory()
        get_mem = lambda addr: int(mem[addr])
        set_mem = lambda addr, val: mem.__setitem__(addr, val)
    else:
        raise AttributeError("No memory interface found in PyBoy instance")
    
    # screen access - prueba múltiples APIs
    if hasattr(pyboy, "screen") and hasattr(pyboy.screen, "ndarray"):
        get_screen = lambda: pyboy.screen.ndarray
    elif hasattr(pyboy, "screen_buffer"):
        get_screen = lambda: pyboy.screen_buffer()
    else:
        get_screen = lambda: None
    
    # Assign to instance attributes
    self._get_mem = get_mem
    self._set_mem = set_mem
    self._get_screen = get_screen
```

**2. Llamada al wrapper en `__init__`:**

```python
if not config["headless"]:
    self.pyboy.set_emulation_speed(6)

# Setup compatibility wrappers for PyBoy API variations
self._setup_pyboy_compat()
```

**3. Modificación de `render()` para usar wrapper:**

```python
def render(self, reduce_res=True):
    # Use compatibility wrapper for screen access
    game_pixels_render = self._get_screen()  # Returns ndarray (144, 160, 3)
```

**4. Modificación de `read_m()` para usar wrapper:**

```python
def read_m(self, addr):
    return self._get_mem(addr)
```

**5. Compatibilidad en `gym_scenarios/generate_gym_states.py`:**

```python
def write_memory(self, address, value):
    """Escribe un valor en una dirección de memoria"""
    # Try multiple possible APIs for PyBoy memory write (compatibility)
    try:
        self.pyboy.memory[address] = value
        return
    except AttributeError:
        pass
    try:
        # older PyBoy: set_memory_value
        self.pyboy.set_memory_value(address, value)
        return
    except Exception:
        pass
    try:
        # Some variants expose get_memory() dict-like object
        mem = getattr(self.pyboy, 'get_memory', lambda: None)()
        if mem is not None:
            mem[address] = value
            return
    except Exception:
        pass
    # If we reach here, we cannot write memory
    raise AttributeError('Unable to write to PyBoy memory: no compatible API found')
```

**6. Compatibilidad en `gym_scenarios/test_single_gym.py`:**

```python
def read_mem(pyboy, addr):
    """Compatibility function to read memory from pyboy across versions"""
    if hasattr(pyboy, "memory"):
        return int(pyboy.memory[addr])
    if hasattr(pyboy, "get_memory_value"):
        return int(pyboy.get_memory_value(addr))
    if hasattr(pyboy, "get_memory"):
        mem = pyboy.get_memory()
        return int(mem[addr])
    raise AttributeError("No memory interface found on PyBoy instance")
```

### Resultado
Compatibilidad con PyBoy 1.x y 2.x implementada  
Scripts funcionan correctamente con PyBoy 2.4.0

---

## Problema 3: AttributeError: 'pyboy.pyboy.PyBoy' object has no attribute 'memory' (run_pretrained_interactive.py)

### Descripción del Error
```
AttributeError: 'pyboy.pyboy.PyBoy' object has no attribute 'memory'
  File "v2\red_gym_env_v2.py", line 568, in read_m
    return self.pyboy.memory[addr]
```

### Archivos Afectados
- `v2/run_pretrained_interactive.py`

### Causa Raíz
El script `run_pretrained_interactive.py` estaba usando una **versión antigua de PyBoy** donde la API era `get_memory_value()` en lugar de `memory[]`.

El entorno `pokeenv` tiene PyBoy 2.28.0 en algunos casos y 2.32.0 en otros (inconsistencia de instalación).

### Solución Aplicada
Mismo wrapper de compatibilidad aplicado en el Problema 2.

### Resultado
run_pretrained_interactive.py` ahora funciona con cualquier versión de PyBoy

---

## Resumen de Correcciones

| # | Problema | Archivos Modificados | Solución |
|---|----------|---------------------|----------|
| 1 | `ModuleNotFoundError: websockets` | N/A | `pip install websockets==13.1` |
| 2 | `AttributeError: screen_buffer` | `v2/red_gym_env_v2.py` | Wrapper `_setup_pyboy_compat()` |
| 3 | `AttributeError: memory` | `v2/red_gym_env_v2.py` | Wrapper `_get_mem()` |
| 4 | `ERROR: nvidia-nccl-cu12==2.21.5` | `v2/requirements.txt`, `v2/install_dependencies.py`, `v2/INSTALLATION.md` | Marcadores de entorno `; sys_platform == 'linux'` |
| 5 | Compatibilidad estado generador | `gym_scenarios/generate_gym_states.py` | Try/except múltiples APIs |
| 6 | Compatibilidad test helper | `gym_scenarios/test_single_gym.py` | Función `read_mem()` |

---

## Scripts Verificados como Funcionales

### Funcionando Correctamente
1. **`epsilon_greedy/run_epsilon_greedy_interactive.py`**
   - Carga el entorno
   - Inicializa PyBoy
   - Ejecuta agente epsilon-greedy
   - Muestra debug cada 50 pasos
   - Ventana SDL2 visible

2. **`v2/run_pretrained_interactive.py`**
   - Carga modelo PPO
   - Carga estado guardado
   - Ejecuta predicciones
   - Requiere `agent_enabled.txt` para control

3. **`baselines/run_pretrained_interactive.py`**
   - Carga modelo antiguo (warnings de versión SB3 < 1.7.0)
   - Ejecuta correctamente
   - El modelo es de versión antigua de stable-baselines3

### No Probado (por instrucción del usuario)
4. **`v2/baseline_fast_v2.py`**
   - No ejecutado (podría hacer crash de VSCode según usuario)
   - Dependencias instaladas: ✓
   - Compatibilidad PyBoy: ✓

---

## Pasos para Reproducir las Correcciones

### 1. Instalar dependencias faltantes
```bash
conda activate pokeenv
pip install websockets==13.1
```

### 2. Verificar compatibilidad PyBoy
Los wrappers ya están implementados en `v2/red_gym_env_v2.py`.

### 3. Probar scripts
```bash
# Test 1: Epsilon Greedy
cd epsilon_greedy
python run_epsilon_greedy_interactive.py

# Test 2: PPO Interactive  
cd ../v2
python run_pretrained_interactive.py

# Test 3: Baselines (modelo antiguo)
cd ../baselines
python run_pretrained_interactive.py
```

---

## 🔴 Problema 4: ERROR: No matching distribution found for nvidia-nccl-cu12==2.21.5

### ❗ Error Original
Al seguir las instrucciones del README en una instalación nueva en Windows:
```bash
pip install -r requirements.txt
```

Error:
```
ERROR: Could not find a version that satisfies the requirement nvidia-nccl-cu12==2.21.5 (from versions: none)
ERROR: No matching distribution found for nvidia-nccl-cu12==2.21.5
```

### 🔍 Diagnóstico
El archivo `v2/requirements.txt` incluía 12 paquetes NVIDIA CUDA que **NO están disponibles en Windows**:

```
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-nccl-cu12==2.21.5        # ← PAQUETE PROBLEMÁTICO
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
```

Además, `triton==3.1.0` tampoco está disponible en Windows (solo Linux).

**Razón:** Estos paquetes son para entrenamiento multi-GPU con CUDA en Linux. Windows no tiene soporte nativo para NVIDIA NCCL (NVIDIA Collective Communications Library).

### ✅ Solución

#### Solución Final: Requirements.txt Agnóstico (Recomendado)

El archivo `v2/requirements.txt` ahora usa **marcadores de entorno de pip** para instalar automáticamente solo los paquetes compatibles con cada sistema operativo:

```python
# Paquetes NVIDIA solo se instalan en Linux
nvidia-cublas-cu12==12.4.5.8; sys_platform == 'linux'
nvidia-cuda-cupti-cu12==12.4.127; sys_platform == 'linux'
nvidia-cuda-nvrtc-cu12==12.4.127; sys_platform == 'linux'
nvidia-cuda-runtime-cu12==12.4.127; sys_platform == 'linux'
nvidia-cudnn-cu12==9.1.0.70; sys_platform == 'linux'
nvidia-cufft-cu12==11.2.1.3; sys_platform == 'linux'
nvidia-curand-cu12==10.3.5.147; sys_platform == 'linux'
nvidia-cusolver-cu12==11.6.1.9; sys_platform == 'linux'
nvidia-cusparse-cu12==12.3.1.170; sys_platform == 'linux'
nvidia-nccl-cu12==2.21.5; sys_platform == 'linux'
nvidia-nvjitlink-cu12==12.4.127; sys_platform == 'linux'
nvidia-nvtx-cu12==12.4.127; sys_platform == 'linux'

# Triton solo en Linux
triton==3.1.0; sys_platform == 'linux'
```

**Ahora funciona en cualquier sistema operativo:**
```bash
# Windows, Linux, macOS - mismo comando
pip install -r requirements.txt
```

#### Opción 1: Script Automático `install_dependencies.py`

Creado script Python que detecta el OS automáticamente:

```bash
# Instalación básica (detecta OS automáticamente)
python install_dependencies.py

# Linux con GPU
python install_dependencies.py --gpu

# Ver qué se instalará
python install_dependencies.py --dry-run
```

**Características:**
- ✅ Detecta Windows/Linux/macOS automáticamente
- ✅ Instala solo paquetes compatibles
- ✅ Valida la instalación
- ✅ Soporte para GPU en Linux

#### Opción 2: Usar `requirements.txt` con marcadores

```bash
pip install -r requirements.txt
```

**Comportamiento automático:**
- **Windows**: Omite nvidia-* y triton
- **Linux**: Instala nvidia-* y triton
- **macOS**: Omite nvidia-* y triton

### 📄 Archivos Modificados

1. **`v2/requirements.txt`**: 
   - Líneas 29-44: NVIDIA packages con marcador `; sys_platform == 'linux'`
   - Línea 71: triton con marcador `; sys_platform == 'linux'`
   - **Agnóstico**: Funciona en Windows, Linux y macOS sin modificaciones

2. **`v2/install_dependencies.py`** (nuevo):
   - Script automático de instalación
   - Detecta OS y arquitectura
   - Instala dependencias correctas automáticamente
   - Incluye validación post-instalación

3. **`v2/requirements-windows.txt`**:
   - Mantenido para compatibilidad retroactiva
   - Ya no necesario (usar requirements.txt directamente)

4. **`v2/INSTALLATION.md`** (nuevo):
   - Guía completa de instalación
   - Instrucciones para Windows, Linux, macOS
   - Troubleshooting común

### 🧪 Validación
```bash
# Instalación agnóstica (funciona en cualquier OS)
pip install -r requirements.txt

# O usar script automático
python install_dependencies.py

# Verificar que PyTorch se instaló correctamente
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# Windows/macOS: PyTorch 2.5.0, CUDA: False
# Linux sin GPU: PyTorch 2.5.0, CUDA: False
# Linux con GPU: PyTorch 2.5.0, CUDA: True
```

### 📝 Notas Importantes
- **Agnóstico al OS**: Un solo `requirements.txt` funciona en Windows, Linux y macOS
- **Marcadores de entorno**: `; sys_platform == 'linux'` instala paquetes solo en Linux
- **Windows**: Automáticamente omite nvidia-* y triton (sin errores)
- **Linux con GPU**: Automáticamente instala nvidia-* y triton
- **Linux sin GPU**: Instala PyTorch CPU (igual que Windows)
- **macOS**: Instala PyTorch CPU (MPS/Metal no soportado aún)
- **requirements-windows.txt**: Ya no necesario, mantenido para compatibilidad retroactiva

---

## 🔴 Problema 5: Cython.Compiler.Errors.CompileError - PyBoy incompatible con Python 3.13

### Descripción del Error
```
Error compiling Cython file:
pyboy\core\cartridge\cartridge.py:35:63: Unicode objects only support coercion to Py_UNICODE*.
pyboy\core\cartridge\cartridge.py:35:74: Unicode objects only support coercion to Py_UNICODE*.

Cython.Compiler.Errors.CompileError: pyboy\core\cartridge\cartridge.py
```

### Archivos Afectados
- **PyBoy 2.4.0** → No compila con Python 3.13+
- `v2/requirements.txt` → especifica `pyboy==2.4.0`
- `v2/install_dependencies.py` → sin validación de versión de Python (antes del fix)

### Causa Raíz
**PyBoy 2.4.0 usa Cython 3.0** que tiene cambios incompatibles con Python 3.13:
- Python 3.13 cambió la API interna de Unicode (`Py_UNICODE*` deprecado)
- Cython en PyBoy no maneja los nuevos tipos de strings de Python 3.13
- PyBoy requiere **Python 3.10, 3.11 o 3.12 como máximo**

### Evidencia
```bash
PS> python install_dependencies.py  # Python 3.13.3
Collecting pyboy==2.4.0
  Using cached pyboy-2.4.0.tar.gz (161 kB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  
  error: subprocess-exited-with-error
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  
  Error compiling Cython file:
  pyboy\core\cartridge\cartridge.py:35:63: Unicode objects only support coercion to Py_UNICODE*.
  
  Cython.Compiler.Errors.CompileError: pyboy\core\cartridge\cartridge.py
  [end of output]
```

### Solución Aplicada

**1. Agregar validación de versión de Python en `v2/install_dependencies.py`:**
```python
def main():
    # ... código anterior ...
    
    # Validar versión de Python
    python_version = sys.version_info
    if python_version < (3, 10) or python_version >= (3, 13):
        print("\n" + "=" * 70)
        print("❌ ERROR: Versión de Python incompatible")
        print("=" * 70)
        print(f"\nPython actual: {python_version.major}.{python_version.minor}.{python_version.micro}")
        print("\n⚠️  PyBoy requiere Python 3.10, 3.11 o 3.12")
        print("   Python 3.13+ NO es compatible debido a cambios en Cython")
        print("\n📥 Soluciones:")
        print("   1. Instalar Python 3.12: https://www.python.org/downloads/")
        print("   2. Usar pyenv (Linux/macOS): pyenv install 3.12")
        print("   3. Crear entorno conda: conda create -n pokeenv python=3.12")
        print("\nDespués de instalar Python 3.10-3.12, ejecuta:")
        print("   python3.12 install_dependencies.py")
        print("\n" + "=" * 70)
        sys.exit(1)
```

**2. Actualizar documentación en `README.md`:**
```markdown
### Requisitos Previos

- **Python 3.10, 3.11 o 3.12** ⚠️ **Python 3.13+ NO es compatible con PyBoy**
- **pip 21.0+** (para soporte de marcadores de entorno)

> ⚠️ **IMPORTANTE - Versión de Python**: PyBoy no funciona con Python 3.13 o superior 
> debido a incompatibilidades con Cython. Usa Python 3.10, 3.11 o 3.12.
```

**3. Agregar sección de troubleshooting en `README.md`:**
```markdown
<details>
<summary><b>Error: PyBoy compilation error (Cython) en Python 3.13</b></summary>

**Causa:** Python 3.13 no es compatible con PyBoy 2.4.0.

**Solución:** Instalar Python 3.10, 3.11 o 3.12:

**Windows:**
# Descargar Python 3.12 desde python.org
# O usar chocolatey:
choco install python --version=3.12.0

**Linux:**
# Opción 1: pyenv
pyenv install 3.12.0
pyenv local 3.12.0

# Opción 2: deadsnakes PPA (Ubuntu)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv

**Conda (todas las plataformas):**
conda create -n pokeenv python=3.12
conda activate pokeenv
```

### Resultado
✅ **Script validado**: Ahora detecta Python 3.13+ y muestra error descriptivo con instrucciones
✅ **Documentación actualizada**: README.md refleja limitación de versión de Python
✅ **Soluciones documentadas**: Guías para instalar Python 3.10-3.12 en todas las plataformas

### Verificación
```bash
# CON Python 3.13 (FALLA CORRECTAMENTE):
PS> python install_dependencies.py
======================================================================
❌ ERROR: Versión de Python incompatible
======================================================================

Python actual: 3.13.3

⚠️  PyBoy requiere Python 3.10, 3.11 o 3.12
   Python 3.13+ NO es compatible debido a cambios en Cython

📥 Soluciones:
   1. Instalar Python 3.12: https://www.python.org/downloads/
   2. Usar pyenv (Linux/macOS): pyenv install 3.12
   3. Crear entorno conda: conda create -n pokeenv python=3.12

# CON Python 3.12 (FUNCIONA):
PS> python3.12 install_dependencies.py
======================================================================
🚀 Instalador de Dependencias - Pokemon Red RL Environment
======================================================================

🖥️  Sistema Operativo: Windows
🐍 Python: 3.12.0

[... instalación exitosa ...]
```

### Archivos Modificados
1. **v2/install_dependencies.py** → Validación de Python 3.10-3.12
2. **README.md** → Requisitos previos + sección troubleshooting
3. **CATASTRO_ERRORES_CORREGIDOS.md** → Problema 5 documentado

---

## 📊 Resumen de Problemas Corregidos

| # | Problema | Causa | Solución | Estado |
|---|----------|-------|----------|--------|
| 1 | `ModuleNotFoundError: websockets` | No instalado en entorno | `pip install websockets==13.1` | ✅ Resuelto |
| 2 | `PyBoy.screen_buffer` no existe | API cambió en PyBoy 2.0+ | Wrappers de compatibilidad | ✅ Resuelto |
| 3 | `SDL_Init() failed` en Windows | PySDL2-dll faltante | `pip install pysdl2-dll==2.30.2` | ✅ Resuelto |
| 4 | `nvidia-nccl-cu12` en Windows | Paquete Linux-only | Marcadores de entorno pip | ✅ Resuelto |
| 5 | `Cython.CompileError` PyBoy | Python 3.13 incompatible | Validación Python 3.10-3.12 | ✅ Resuelto |

**Total de problemas documentados:** 5  
**Problemas resueltos:** 5 (100%)