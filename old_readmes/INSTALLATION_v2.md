# 📦 Guía de Instalación - Pokemon Red RL Environment

Esta guía proporciona instrucciones de instalación **agnósticas al sistema operativo** para Windows, Linux y macOS.

---

## Instalación Rápida (Recomendada)

### Método 1: Script Automático (Recomendado)

El script `install_dependencies.py` detecta automáticamente tu sistema operativo e instala las dependencias correctas:

```bash
# Instalación básica (CPU)
python install_dependencies.py

# Linux con GPU CUDA (opcional)
python install_dependencies.py --gpu

# Ver qué se instalará sin instalar
python install_dependencies.py --dry-run
```

**Ventajas:**
- Detecta automáticamente Windows/Linux/macOS
- Instala solo las dependencias compatibles
- Valida la instalación automáticamente
- No requiere editar archivos manualmente

---

### Método 2: pip install requirements.txt

El archivo `requirements.txt` usa **marcadores de entorno** de pip para instalar automáticamente solo los paquetes compatibles:

```bash
pip install -r requirements.txt
```

**Cómo funciona:**
- En **Windows**: Omite automáticamente paquetes NVIDIA y triton
- En **Linux**: Instala paquetes NVIDIA CUDA y triton para GPU
- En **macOS**: Omite paquetes incompatibles

**Marcadores de entorno usados:**
```python
nvidia-nccl-cu12==2.21.5; sys_platform == 'linux'  # Solo en Linux
triton==3.1.0; sys_platform == 'linux'              # Solo en Linux
```

---

## Instalación Manual Paso a Paso

### Windows

```bash
# 1. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
venv\Scripts\activate

# 2. Actualizar pip
python -m pip install --upgrade pip

# 3. Instalar dependencias (automáticamente omite NVIDIA packages)
pip install -r requirements.txt

# 4. Verificar instalación
python -c "import torch; import pyboy; import gymnasium; print('✅ Todo instalado')"
```

### Linux (CPU)

```bash
# 1. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 2. Actualizar pip
pip install --upgrade pip

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Linux (GPU con CUDA)

```bash
# 1. Verificar que tienes CUDA 12.4 instalado
nvidia-smi

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias (incluye NVIDIA packages automáticamente)
pip install -r requirements.txt

# 4. Verificar GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### macOS

```bash
# 1. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# Nota: macOS usa PyTorch CPU (MPS/Metal no soportado aún en este proyecto)
```

---

## Verificación de Instalación

Después de instalar, verifica que todo funciona:

```python
# Test completo
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
print('Websockets instalado')
print('\n¡Todas las dependencias instaladas correctamente!')
"
```

---

## Solución de Problemas

### Error: "No matching distribution found for nvidia-nccl-cu12"

**Causa:** Estás en Windows o macOS intentando instalar paquetes NVIDIA.

**Solución:**
```bash
# Asegúrate de usar requirements.txt actualizado con marcadores de entorno
pip install -r requirements.txt

# O usa el script automático
python install_dependencies.py
```

### Error: "ModuleNotFoundError: No module named 'websockets'"

**Solución:**
```bash
pip install websockets==13.1
```

### Error: PyBoy API incompatibilidad

**Solución:** Los wrappers de compatibilidad ya están implementados en `red_gym_env_v2.py`. Si sigues teniendo problemas:

```bash
# Reinstalar PyBoy versión específica
pip install pyboy==2.4.0 --force-reinstall
```

### Windows: SDL2 DLL no encontrado

**Solución:**
```bash
pip install pysdl2-dll==2.30.2 --force-reinstall
```

---

## 🎮 Opciones Específicas de Plataforma

### Windows con GPU (NVIDIA)

PyTorch con CUDA en Windows requiere instalación manual:

```bash
# Desinstalar PyTorch CPU
pip uninstall torch

# Instalar PyTorch con CUDA 11.8 (más compatible en Windows)
pip install torch==2.5.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### Linux con CUDA 11.x en vez de 12.x

Si tienes CUDA 11.x instalado en vez de 12.x:

```bash
# Modificar requirements.txt para usar cu118 en vez de cu124
pip install torch==2.5.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Los paquetes nvidia-*-cu12 no funcionarán, usa:
pip install -r requirements.txt --no-deps  # Omite dependencias
pip install torch==2.5.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Comparación de Métodos

| Método | Windows | Linux CPU | Linux GPU | macOS | Automático |
|--------|---------|-----------|-----------|-------|------------|
| `install_dependencies.py` | Si | Si | Si | Si | Si |
| `requirements.txt` | Si | Si | Si | Si | Si |
| `requirements-windows.txt` | Si | No | No | No | No |

**Recomendación:** Usar `install_dependencies.py` o `requirements.txt` para máxima compatibilidad.

---

## Requisitos de Sistema

- **Python:** 3.10 o superior
- **pip:** 21.0 o superior
- **Sistema Operativo:** Windows 10+, Linux (Ubuntu 20.04+, Debian 11+), macOS 11+
- **RAM:** Mínimo 4GB (recomendado 8GB para entrenamiento)
- **GPU (opcional):** NVIDIA con CUDA 12.4 (solo Linux)

---

## Recursos Adicionales

- **PyTorch:** https://pytorch.org/get-started/locally/
- **PyBoy:** https://github.com/Baekalfen/PyBoy
- **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/
- **Troubleshooting:** Ver `CATASTRO_ERRORES_CORREGIDOS.md`

---

## Siguiente Paso

Una vez instaladas las dependencias:

```bash
# Ejecutar agente interactivo
python run_pretrained_interactive.py

# O ejecutar entrenamiento
python baseline_fast_v2.py
```

