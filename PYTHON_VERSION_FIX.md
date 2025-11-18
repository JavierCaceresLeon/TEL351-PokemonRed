# 🐍 Guía de Instalación de Python 3.12 para PyBoy

> **Problema:** PyBoy 2.4.0 no es compatible con Python 3.13 debido a cambios en Cython.  
> **Solución:** Instalar Python 3.10, 3.11 o 3.12.

---

## 🔍 Verificar Versión Actual

```bash
python --version
```

Si muestra `Python 3.13.x`, necesitas instalar Python 3.12.

---

## 💻 Windows

### Opción 1: Instalador Oficial (Más Fácil)

1. **Descargar Python 3.12:**
   - Ir a https://www.python.org/downloads/
   - Descargar **Python 3.12.8** (última versión estable)
   - Elegir "Windows installer (64-bit)"

2. **Instalar:**
   - ✅ Marcar "Add Python 3.12 to PATH"
   - Click "Install Now"
   - Esperar instalación

3. **Verificar:**
   ```bash
   python --version
   # Debe mostrar: Python 3.12.8
   ```

4. **Usar Python 3.12 específicamente:**
   ```bash
   # Si tienes múltiples versiones:
   py -3.12 --version
   py -3.12 install_dependencies.py
   ```

### Opción 2: Chocolatey

```bash
# Instalar Chocolatey primero (si no lo tienes):
# https://chocolatey.org/install

# Instalar Python 3.12:
choco install python --version=3.12.8

# Verificar:
python --version
```

### Opción 3: Conda (Recomendado para Desarrollo)

```bash
# Crear entorno con Python 3.12:
conda create -n pokeenv python=3.12

# Activar entorno:
conda activate pokeenv

# Verificar:
python --version  # Python 3.12.x

# Navegar a proyecto e instalar:
cd C:\Users\javi1\Documents\repos_git\TEL351-PokemonRed\v2
python install_dependencies.py
```

---

## 🐧 Linux

### Opción 1: pyenv (Recomendado)

```bash
# Instalar pyenv (si no lo tienes):
curl https://pyenv.run | bash

# Agregar a ~/.bashrc o ~/.zshrc:
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Recargar shell:
source ~/.bashrc

# Instalar Python 3.12:
pyenv install 3.12.8

# Establecer como versión local (solo en este proyecto):
cd ~/TEL351-PokemonRed
pyenv local 3.12.8

# Verificar:
python --version  # Python 3.12.8
```

### Opción 2: deadsnakes PPA (Ubuntu/Debian)

```bash
# Agregar PPA:
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Instalar Python 3.12:
sudo apt install python3.12 python3.12-venv python3.12-dev

# Crear entorno virtual:
python3.12 -m venv pokeenv

# Activar:
source pokeenv/bin/activate

# Verificar:
python --version  # Python 3.12.x
```

### Opción 3: Conda

```bash
# Crear entorno con Python 3.12:
conda create -n pokeenv python=3.12

# Activar:
conda activate pokeenv

# Verificar:
python --version  # Python 3.12.x
```

---

## 🍎 macOS

### Opción 1: pyenv (Recomendado)

```bash
# Instalar Homebrew (si no lo tienes):
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalar pyenv:
brew install pyenv

# Agregar a ~/.zshrc o ~/.bash_profile:
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Recargar shell:
source ~/.zshrc

# Instalar Python 3.12:
pyenv install 3.12.8

# Establecer como versión local:
cd ~/TEL351-PokemonRed
pyenv local 3.12.8

# Verificar:
python --version  # Python 3.12.8
```

### Opción 2: Homebrew Directo

```bash
# Instalar Python 3.12:
brew install python@3.12

# Verificar:
python3.12 --version

# Crear alias (opcional):
echo 'alias python=python3.12' >> ~/.zshrc
source ~/.zshrc
```

### Opción 3: Conda

```bash
# Instalar Miniconda:
brew install --cask miniconda

# Crear entorno:
conda create -n pokeenv python=3.12
conda activate pokeenv

# Verificar:
python --version  # Python 3.12.x
```

---

## 🚀 Después de Instalar Python 3.12

### Windows:

```bash
# Navegar al proyecto:
cd C:\Users\javi1\Documents\repos_git\TEL351-PokemonRed\v2

# Ejecutar instalador:
python install_dependencies.py

# O si tienes múltiples versiones:
py -3.12 install_dependencies.py
```

### Linux/macOS:

```bash
# Navegar al proyecto:
cd ~/TEL351-PokemonRed/v2

# Ejecutar instalador:
python install_dependencies.py

# O especificar versión:
python3.12 install_dependencies.py
```

---

## ✅ Verificar Instalación Correcta

```bash
# Verificar versión de Python:
python --version
# Debe mostrar: Python 3.12.x (donde x puede ser 0-8)

# Ejecutar script de instalación:
python install_dependencies.py

# Si todo está bien, deberías ver:
======================================================================
🚀 Instalador de Dependencias - Pokemon Red RL Environment
======================================================================

🖥️  Sistema Operativo: Windows
⚙️  Arquitectura: AMD64
🐍 Python: 3.12.8

🔧 Modo de instalación:
   • Windows: PyTorch CPU

¿Continuar con la instalación? [S/n]:
```

---

## ⚠️ Solución de Problemas

### Error: "python: command not found" después de instalar

**Windows:**
```bash
# Usar py launcher:
py -3.12 --version

# O agregar a PATH manualmente:
# Panel de Control → Sistema → Configuración avanzada del sistema → Variables de entorno
# Agregar: C:\Users\<TU_USUARIO>\AppData\Local\Programs\Python\Python312
```

**Linux/macOS:**
```bash
# Recargar configuración de shell:
source ~/.bashrc   # Linux
source ~/.zshrc    # macOS

# O usar ruta completa:
~/.pyenv/versions/3.12.8/bin/python --version
```

### Error: "Multiple Python versions conflict"

**Solución:** Usar entorno virtual o conda:

```bash
# Opción 1: venv (Python estándar)
python3.12 -m venv pokeenv
source pokeenv/bin/activate  # Linux/macOS
pokeenv\Scripts\activate     # Windows

# Opción 2: conda
conda create -n pokeenv python=3.12
conda activate pokeenv
```

---

## 🔗 Recursos Adicionales

- **Python 3.12 Downloads:** https://www.python.org/downloads/
- **pyenv GitHub:** https://github.com/pyenv/pyenv
- **Conda Installation:** https://docs.conda.io/projects/conda/en/latest/user-guide/install/
- **PyBoy GitHub:** https://github.com/Baekalfen/PyBoy/issues (verificar compatibilidad)

---

## 📝 Notas Importantes

- ✅ **Python 3.10, 3.11, 3.12:** Totalmente compatibles con PyBoy 2.4.0
- ❌ **Python 3.13+:** NO compatible (errores de Cython)
- ⚠️ **Python 3.9 o anterior:** NO recomendado (dependencias antiguas)

**Versión recomendada:** Python 3.12.8 (última estable compatible)
