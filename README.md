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

# 5. Ejecutar modelo preentrenado
python run_pretrained_interactive.py
```

---

## Entrenamiento y Evaluación de Modelos

### Entrenar Desde Cero: baseline_fast_v2.py

El script `baseline_fast_v2.py` permite entrenar un agente PPO (Proximal Policy Optimization) desde cero o continuar entrenamiento desde un checkpoint existente.

**Características principales:**
- Entrenamiento con algoritmo PPO de Stable Baselines3
- Soporte para entrenamiento paralelo multi-CPU
- Guardado automático de checkpoints cada N pasos
- Integración con TensorBoard para visualización de métricas
- Streaming de mapa de exploración en tiempo real

**Ejecución básica:**
```bash
# Activar entorno
conda activate pokeenv

# Navegar a directorio v2
cd v2

# Entrenar desde cero con ventana visible (1 CPU)
python baseline_fast_v2.py
```

**Parámetros configurables:**

Editar directamente en `baseline_fast_v2.py`:

```python
# Líneas 37-46: Configuración del entorno
env_config = {
    'headless': False,          # True = sin ventana, False = ventana visible
    'action_freq': 24,          # Frames por acción (12-48 recomendado)
    'reward_scale': 0.5,        # Escala de recompensas (0.1-1.0)
    'explore_weight': 0.25,     # Peso de exploración (0.0-1.0)
    'max_steps': ep_length,     # Pasos máximos por episodio
    'print_rewards': True,      # Mostrar recompensas en consola
}

# Línea 50: Número de procesos paralelos
num_cpu = 1  # 1 = ventana visible, 4-64 = entrenamiento rápido paralelo

# Línea 37: Longitud de episodios
ep_length = 2048 * 80  # Ajustar multiplicador (20-160)

# Línea 92: Hiperparámetros del algoritmo PPO
model = PPO(
    "MultiInputPolicy", 
    env, 
    verbose=1,
    n_steps=train_steps_batch,  # Pasos antes de actualizar política
    batch_size=512,             # Tamaño del batch (256, 512, 1024)
    n_epochs=1,                 # Épocas de optimización (1-10)
    gamma=0.997,                # Factor de descuento (0.95-0.999)
    ent_coef=0.01,              # Coeficiente de entropía (0.001-0.1)
    tensorboard_log=sess_path
)

# Línea 96: Pasos totales de entrenamiento
total_timesteps = (ep_length) * num_cpu * 10000
```

**Recomendaciones para modificar el entrenamiento:**

1. **Entrenamiento interactivo (ver ventana):**
   - `headless = False`
   - `num_cpu = 1`

2. **Entrenamiento rápido (sin ventana):**
   - `headless = True`
   - `num_cpu = 16` o más (según RAM disponible)

3. **Mayor exploración:**
   - `explore_weight = 0.5`
   - `ent_coef = 0.05`

4. **Aprendizaje más agresivo:**
   - `n_epochs = 3`
   - `batch_size = 256`

5. **Episodios más cortos (pruebas rápidas):**
   - `ep_length = 2048 * 20`
   - `total_timesteps = ep_length * 100`

**Checkpoints y continuación de entrenamiento:**

Los checkpoints se guardan automáticamente en la carpeta `runs/` cada `ep_length//2` pasos. Para continuar desde un checkpoint:

```bash
# Redirigir stdin con el nombre del checkpoint
echo "runs/poke_26214400_steps" | python baseline_fast_v2.py

# O modificar línea 82 del archivo directamente:
file_name = "runs/poke_26214400_steps"
```
---

### Evaluar Modelo Entrenado: run_ppo_interactive_metrics.py

El script `run_ppo_interactive_metrics.py` ejecuta un modelo PPO preentrenado y genera métricas detalladas de rendimiento.

**Características:**
- Carga automática del checkpoint más reciente desde `runs/`
- Ejecución con ventana de Game Boy visible
- Captura de métricas en tiempo real (recompensas, acciones, recursos del sistema)
- Generación de reportes completos en Markdown, JSON y CSV

**Ejecución:**
```bash
# Activar entorno
conda activate pokeenv

# Navegar a v2
cd v2

# Ejecutar modelo preentrenado
python run_ppo_interactive_metrics.py

# El script automáticamente carga el .zip más reciente de la carpeta runs/
# Presionar Ctrl+C para detener y generar métricas
```

**Modelo utilizado:**

El script carga automáticamente el checkpoint **más reciente** (archivo `.zip`) encontrado en la carpeta `runs/`. Por ejemplo:
- `runs/poke_81920_steps.zip`
- `runs/poke_163840_steps.zip`
- `runs/poke_26214400_steps.zip`

Si la carpeta `runs/` no contiene checkpoints, el script mostrará un error. Asegúrate de tener un modelo entrenado antes de ejecutar, que ya se haya ejecutado antes para que se haya generado un archivo en `runs/`.

**Salida generada:**

Al finalizar la ejecución (por Ctrl+C o completar episodio), se crean tres archivos en `ppo_results/`:

1. **`ppo_metrics_[timestamp].md`** - Reporte detallado en Markdown con:
   - Rendimiento principal (recompensas totales, máximas, mínimas)
   - Análisis temporal (pasos/segundo, tiempo total)
   - Información del modelo (algoritmo, checkpoint cargado)
   - Uso de recursos (memoria, CPU)
   - Estadísticas de acciones y recompensas
   - Comportamiento del agente

2. **`ppo_raw_data_[timestamp].json`** - Datos crudos para análisis posterior:
   - Historial de acciones (últimas 1000)
   - Historial de recompensas (últimas 1000)
   - Métricas de rendimiento
   - Información del sistema

3. **`ppo_summary_[timestamp].csv`** - Resumen en formato tabla para análisis estadístico

**Configuración:**

El script usa configuración fija para evaluación:
- Ventana visible (`headless: False`)
- 1 proceso (`num_cpu: 1`)
- Modo no determinístico para capturar variabilidad del modelo

Para modificar el comportamiento, editar líneas 213-225 en el archivo.

---

### Algoritmo Epsilon Greedy: Búsqueda Heurística

El proyecto incluye una implementación alternativa basada en **Epsilon Greedy**, un algoritmo de búsqueda heurística que NO requiere entrenamiento previo.

**Diferencias fundamentales con PPO:**

| Aspecto | PPO (Aprendizaje por Refuerzo) | Epsilon Greedy (Búsqueda Heurística) |
|---------|--------------------------------|--------------------------------------|
| Paradigma | Deep Learning / Reinforcement Learning | Algoritmo de búsqueda con heurísticas |
| Entrenamiento | Requiere miles de episodios | NO requiere entrenamiento |
| Funcionamiento | Aprende políticas mediante experiencia | Evalúa acciones con funciones predefinidas |
| Adaptación | Mejora con más datos | Funciona inmediatamente |
| Costo computacional | Alto (GPU recomendada) | Bajo (solo evaluación de funciones) |
| Transparencia | Caja negra (red neuronal) | Completamente explicable (heurísticas) |

**Características del agente Epsilon Greedy:**
- Detección automática de escenarios (exploración, combate, navegación, progresión)
- Heurísticas adaptativas según el contexto del juego
- Pesos diferentes para cada escenario detectado
- Sin dependencia de checkpoints preentrenados
- Comportamiento completamente determinista y reproducible

**Ejecutar Epsilon Greedy Interactivo:**

```bash
# Activar entorno (mismo que v2)
conda activate pokeenv

# Navegar a carpeta raíz del proyecto
cd TEL351-PokemonRed

# Ejecutar agente Epsilon Greedy con ventana visible
python epsilon_greedy/run_epsilon_greedy_interactive.py

# Presionar Ctrl+C para detener y generar métricas
```

**Salida generada:**

Similar a `run_ppo_interactive_metrics.py`, el script genera tres archivos en `epsilon_greedy/results/`:

1. **`epsilon_greedy_metrics_[timestamp].md`** - Reporte detallado con:
   - Rendimiento y recompensas
   - Uso de heurísticas por escenario
   - Detección de escenarios (exploración, combate, menús)
   - Estadísticas de acciones
   - Configuración del agente

2. **`epsilon_greedy_raw_data_[timestamp].json`** - Datos crudos:
   - Historial de acciones y recompensas
   - Uso de heurísticas por tipo
   - Detecciones de escenarios
   - Historial de epsilon (factor de exploración)

3. **`epsilon_greedy_summary_[timestamp].csv`** - Resumen en formato tabla

**Configuración del agente:**

Los parámetros principales se configuran en `epsilon_greedy/epsilon_greedy_agent.py`:

```python
# Parámetros de exploración
epsilon_start = 0.3   # Probabilidad inicial de acción aleatoria (0.0-1.0)
epsilon_min = 0.05    # Probabilidad mínima
epsilon_decay = 0.995 # Factor de decaimiento

# Pesos heurísticos por escenario (en HeuristicWeights)
# EXPLORATION: Mayor peso en exploración y familiaridad del mapa
# BATTLE: Prioriza salud y nivel de Pokémon
# NAVIGATION: Enfoque en distancia a objetivos
# PROGRESSION: Maximiza eventos clave (gimnasios, capturas)
# STUCK: Aumenta exploración para salir de bucles
```

**Ventajas de Epsilon Greedy:**
- Funciona "out of the box" sin entrenamiento previo
- Completamente explicable (cada decisión tiene justificación heurística)
- Generalizable a Pokemon Blue, hacks ROM, modificaciones
- Robusto ante situaciones no vistas previamente
- Menor costo computacional

**Desventajas:**
- Más lento que PPO entrenado (5-10 minutos vs 2-4 minutos)
- Requiere programación manual de heurísticas
- No mejora automáticamente con experiencia

**Uso recomendado:**

Epsilon Greedy es ideal para:
- Investigación de algoritmos de búsqueda
- Comparación con métodos de aprendizaje por refuerzo
- Situaciones donde no se dispone de datos de entrenamiento
- Entornos que cambian frecuentemente (hacks ROM)
- Análisis de comportamiento interpretable

---

### Algoritmo A* (A-Star): Pathfinding Óptimo

El proyecto incluye una implementación de **A*** (A-Star), un algoritmo de búsqueda de caminos que encuentra rutas óptimas hacia objetivos específicos.

**Características de A*:**

| Aspecto | Descripción |
|---------|-------------|
| Paradigma | Algoritmo de búsqueda informada (pathfinding) |
| Optimalidad | Garantiza encontrar el camino más corto |
| Heurística | Usa función h(n) para estimar distancia a objetivo |
| Memoria | Mantiene lista abierta y cerrada de nodos |
| Eficiencia | Evita explorar áreas no prometedoras |

**Funcionamiento de A*:**

El algoritmo usa la función de evaluación: **f(n) = g(n) + h(n)**
- **g(n)**: Costo real desde el inicio hasta el nodo n
- **h(n)**: Heurística (estimación del costo desde n hasta el objetivo)
- **f(n)**: Estimación del costo total del mejor camino que pasa por n

**Ejecutar A* Interactivo:**

```bash
# Activar entorno (mismo que v2)
conda activate pokeenv

# Navegar a carpeta raíz del proyecto
cd TEL351-PokemonRed

# Ejecutar agente A* con ventana visible
python astar_search/run_astar_interactive_simple.py

# Presionar Ctrl+C para detener
```

**Nota importante:** El script debe ejecutarse desde la carpeta raíz del proyecto (TEL351-PokemonRed), NO desde dentro de astar_search/.

**Configuración del agente:**

Parámetros en `run_astar_interactive_simple.py`:

```python
agent_config = {
    'exploration_bonus': 0.2,    # Bonificación por explorar nuevas áreas
    'stuck_threshold': 50,       # Pasos antes de considerar "atascado"
}

env_config = {
    'init_state': '../has_pokedex.state',  # Estado inicial (con Pokédex)
    # ... resto de configuración estándar v2
}
```

**Ventajas de A*:**
- Encuentra caminos óptimos garantizados
- Dirigido por objetivos (no explora aleatoriamente)
- Eficiente en memoria para objetivos específicos
- Predecible y determinista

**Desventajas:**
- Requiere definir objetivos claros
- Mayor complejidad de implementación
- Rendimiento depende de la calidad de la heurística
- Puede ser lento en mapas muy grandes

**Uso recomendado:**

A* es ideal para:
- Navegación a ubicaciones específicas (gimnasios, centros Pokémon)
- Tareas con objetivos bien definidos
- Análisis de rutas óptimas
- Comparación de eficiencia de pathfinding

**Dependencias:**

A* usa el mismo entorno `pokeenv` (no requiere instalación adicional).

---

### Comparación Visual de Agentes: Ejecución Dual Interactiva

La carpeta `comparison_agents/` incluye el script `run_dual_interactive.py` que permite **comparar visualmente** dos algoritmos ejecutándose simultáneamente con ventanas separadas del Game Boy.

**Agentes configurables para comparación:**

Por defecto, el script compara:
1. **Epsilon Greedy** (búsqueda heurística) vs **PPO** (deep learning)

Pero puedes editarlo para comparar cualquier combinación:
- Epsilon Greedy vs A* Search
- Tabu Search vs PPO
- A* vs Epsilon Greedy
- Cualquier combinación de algoritmos disponibles

**Ejecutar comparación dual:**

```bash
# Activar entorno
conda activate pokeenv

# Navegar a carpeta comparison_agents
cd TEL351-PokemonRed/comparison_agents

# Ejecutar comparación dual (por defecto: Epsilon Greedy vs PPO)
python run_dual_interactive.py
```

**Salida esperada:**
- Se abrirán **2 ventanas del Game Boy simultáneamente**
- Ventana 1: Primer agente (por defecto Epsilon Greedy)
- Ventana 2: Segundo agente (por defecto PPO)
- Puedes observar en tiempo real cómo cada algoritmo toma decisiones diferentes

**Detener la comparación:**
- Presiona `Ctrl+C` en la terminal
- Ambos agentes se detendrán
- Las métricas se generarán automáticamente

**Métricas generadas:**

Al presionar `Ctrl+C`, cada agente guarda sus métricas en carpetas separadas:

1. **Epsilon Greedy** → `epsilon_greedy/results/`
   - `epsilon_greedy_metrics_[timestamp].md` - Reporte detallado
   - `epsilon_greedy_raw_data_[timestamp].json` - Datos crudos
   - `epsilon_greedy_summary_[timestamp].csv` - Resumen tabla

2. **PPO** → `v2/ppo_results/`
   - `ppo_metrics_[timestamp].md` - Reporte detallado
   - `ppo_raw_data_[timestamp].json` - Datos crudos
   - `ppo_summary_[timestamp].csv` - Resumen tabla

**Personalizar la comparación:**

Para cambiar qué algoritmos comparar, edita `comparison_agents/run_dual_interactive.py`:

```python
# Ejemplo 1: Comparar Epsilon Greedy vs A* Search
def run_first_agent():
    """Ejecutar primer agente"""
    print("Iniciando Agente Epsilon Greedy...")
    subprocess.run([
        sys.executable, 
        "run_epsilon_greedy_interactive.py"
    ], cwd=Path(__file__).parent.parent / "epsilon_greedy", check=True)

def run_second_agent():
    """Ejecutar segundo agente (A*)"""
    print("Iniciando Agente A*...")
    subprocess.run([
        sys.executable, 
        "run_astar_interactive_simple.py"
    ], cwd=Path(__file__).parent.parent / "astar_search", check=True)

# Ejemplo 2: Comparar Tabu vs Epsilon Greedy
def run_first_agent():
    """Ejecutar Tabu Search"""
    print("Iniciando Tabu Search...")
    subprocess.run([
        sys.executable, 
        "run_tabu_interactive_metrics.py"
    ], cwd=Path(__file__).parent, check=True)

def run_second_agent():
    """Ejecutar Epsilon Greedy"""
    print("Iniciando Epsilon Greedy...")
    subprocess.run([
        sys.executable, 
        "run_epsilon_greedy_interactive.py"
    ], cwd=Path(__file__).parent.parent / "epsilon_greedy", check=True)
```

**Scripts disponibles para comparación:**

| Algoritmo | Script | Carpeta |
|-----------|--------|---------|
| Epsilon Greedy | `run_epsilon_greedy_interactive.py` | `epsilon_greedy/` |
| PPO | `run_pretrained_interactive.py` | `v2/` |
| PPO con métricas | `run_ppo_interactive_metrics.py` | `v2/` |
| A* Search | `run_astar_interactive_simple.py` | `astar_search/` |
| Tabu Search | `run_tabu_interactive_metrics.py` | `comparison_agents/` |

**Ventajas de la comparación dual:**
- Observación visual directa de diferencias entre algoritmos
- Mismas condiciones iniciales para ambos agentes
- Métricas detalladas para análisis posterior
- Fácil de personalizar para cualquier combinación

**Uso recomendado:**

La comparación dual es ideal para:
- Validar mejoras entre algoritmos (versión antigua vs nueva)
- Comparar enfoques (heurísticas vs aprendizaje profundo)
- Demostrar diferencias de comportamiento visualmente
- Generar datos para análisis comparativo posterior

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

<details>
<summary><b>Error: "ModuleNotFoundError: No module named 'v2'" o imports relativos</b></summary>

**Causa:** Problemas con imports de módulos cuando se ejecuta desde diferentes directorios.

**Solución aplicada:** El código ahora usa imports con fallback automático. Si encuentras este error:

1. Verifica que estés ejecutando desde la carpeta correcta:
   ```bash
   # A* Search
   cd TEL351-PokemonRed
   python astar_search/run_astar_interactive_simple.py
   
   # Epsilon Greedy
   cd TEL351-PokemonRed/epsilon_greedy
   python run_epsilon_greedy_interactive.py
   
   # Comparison scripts
   cd TEL351-PokemonRed/comparison_agents
   python run_comparison.py
   ```

2. Si persiste, verifica que exista `v2/__init__.py`

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

**Checkpoints y continuación de entrenamiento:**

Los checkpoints se guardan automáticamente en la carpeta `runs/` cada `ep_length//2` pasos. Para continuar desde un checkpoint:

```bash
# Redirigir stdin con el nombre del checkpoint
echo "runs/poke_26214400_steps" | python baseline_fast_v2.py

# O modificar línea 82 del archivo directamente:
file_name = "runs/poke_26214400_steps"
```
