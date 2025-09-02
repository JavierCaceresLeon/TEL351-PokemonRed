# Entrenamiento de Agentes de Aprendizaje por Refuerzo para Pokémon Red

Este proyecto implementa un entorno de aprendizaje por refuerzo para entrenar agentes de IA que jueguen Pokémon Red automáticamente. El agente aprende a navegar por el mundo del juego, capturar Pokémon, luchar en batallas y completar objetivos usando técnicas de aprendizaje profundo.

## Descripción General del Proyecto

El proyecto utiliza PyBoy (un emulador de Game Boy) junto con Stable Baselines3 para crear un entorno de gimnasio donde los agentes pueden interactuar con Pokémon Red. El agente observa las pantallas del juego y aprende políticas óptimas mediante algoritmos como PPO (Proximal Policy Optimization).

## Instalación y Configuración Rápida

### Requisitos Previos
- Python 3.10+ (recomendado)
- ffmpeg instalado y disponible en la línea de comandos
- ROM de Pokémon Red legalmente obtenida (1MB, sha1: `ea9bcae617fdf159b045185467ae58b2e4a48b9a`)

### Pasos de Instalación

1. **Clonar el repositorio original:**
   ```bash
   git clone https://github.com/PWhiddy/PokemonRedExperiments.git
   cd PokemonRedExperiments
   ```

2. **Crear y activar entorno conda:**
   ```bash
   conda create -n pokeenv python=3.10
   conda activate pokeenv
   ```

3. **Actualizar herramientas de Python:**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

4. **Colocar la ROM de Pokémon Red:**
   - Copiar la ROM legalmente obtenida al directorio base
   - Renombrarla a `PokemonRed.gb`
   - Verificar con: `shasum PokemonRed.gb` (debe coincidir con el hash mencionado arriba)

5. **Instalar dependencias:**
   ```bash
   cd baselines
   pip install -r requirements.txt
   ```
   Para macOS con V2: usar `v2/macos_requirements.txt`

6. **Ejecutar el modelo preentrenado:**
   ```bash
   python run_pretrained_interactive.py
   ```

## Estructura del Proyecto y Descripción de Archivos

### Directorio Raíz
```
TEL351-PokemonRed/
├── PokemonRed.gb              # ROM del juego (debe ser proporcionada por el usuario)
├── README.md                  # Este archivo de documentación
├── README_BASE.md             # Documentación original del proyecto
├── LICENSE                    # Licencia del proyecto
├── windows-setup-guide.md     # Guía específica para Windows
├── VisualizeProgress.ipynb    # Notebook para visualizar el progreso del entrenamiento
└── *.state                    # Estados guardados del juego
```

#### Estados del Juego (*.state)
- **`init.state`**: Estado inicial básico del juego
- **`has_pokedex.state`**: Estado donde el jugador ya tiene la Pokédex
- **`has_pokedex_nballs.state`**: Estado con Pokédex y Pokéballs
- **`fast_text_start.state`**: Estado optimizado para texto rápido

### Directorio `baselines/` (Versión Original)

**Archivos principales de entrenamiento:**
- **`red_gym_env.py`**: **ARCHIVO CLAVE** - Define el entorno de gimnasio principal donde el agente interactúa con el juego
- **`run_baseline_parallel.py`**: Script para entrenar múltiples agentes en paralelo
- **`run_baseline_parallel_fast.py`**: Versión optimizada del entrenamiento paralelo
- **`run_pretrained_interactive.py`**: **EJECUTAR MODELO** - Script para ejecutar el modelo preentrenado de forma interactiva

**Archivos de configuración del agente:**
- **`memory_addresses.py`**: **CONFIGURACIÓN DEL JUEGO** - Define las direcciones de memoria para acceder a datos del juego (posición, salud, dinero, etc.)
- **`agent_enabled.txt`**: Archivo de control para pausar/reanudar la IA durante la ejecución
- **`global_map.py`**: Manejo del mapa global del juego
- **`map_data.json`**: Datos del mapa en formato JSON

**Archivos de utilidades:**
- **`stream_agent_wrapper.py`**: Wrapper para transmitir sesiones de entrenamiento en vivo
- **`tensorboard_callback.py`**: Callbacks para logging con TensorBoard
- **`tile_vids_to_grid.py`**: Utilidad para crear videos en cuadrícula
- **`render_all_needed_grids.py`**: Renderizado de cuadrículas necesarias
- **`delete_empty_imgs.txt`**: Script para limpiar imágenes vacías

**Archivos de datos:**
- **`requirements.txt`**: Dependencias de Python requeridas
- **`events.json`**: Eventos del juego en formato JSON
- **`saves_to_record.txt`**: Lista de estados guardados para grabar

### Directorio `v2/` (Versión Mejorada - Recomendada)

**Mejoras de la V2:**
- Entrenamiento más rápido y eficiente en memoria
- Alcanza Cerulean City
- Streaming al mapa habilitado por defecto
- Recompensa de exploración basada en coordenadas en lugar de KNN de frames

**Archivos principales:**
- **`red_gym_env_v2.py`**: **ENTORNO MEJORADO** - Versión optimizada del entorno de gimnasio
- **`baseline_fast_v2.py`**: **ENTRENAMIENTO V2** - Script principal de entrenamiento de la versión 2
- **`run_pretrained_interactive.py`**: Ejecutor del modelo preentrenado para V2
- **`requirements.txt`** / **`macos_requirements.txt`**: Dependencias específicas para cada SO

### Directorio `visualization/`

**Notebooks y scripts de visualización:**
- **`Agent_Visualization.ipynb`**: Visualización del comportamiento del agente
- **`BetterMapVis.ipynb`**: Visualización mejorada del mapa
- **`BetterMapVis_script_version.py`**: Versión en script de la visualización del mapa
- **`BetterMapVis_script_version_FLOW.py`**: Visualización con flujo de movimiento
- **`Create_Video_Grids.ipynb`**: Creación de videos en cuadrícula
- **`Map_Stitching.ipynb`**: Unión de mapas
- **`MapWalkingVis.ipynb`**: Visualización de caminatas en el mapa

### Directorio `clip_experiment/`

Experimentos con CLIP (Contrastive Language-Image Pre-training):
- **`Interacting_with_CLIP_Pokemon.ipynb`**: Notebook para experimentos con CLIP
- **`location_descriptions/`**: Imágenes con descripciones de ubicaciones
- **`test_images/`**: Imágenes de prueba para CLIP

### Directorio `assets/`

Recursos gráficos y multimedia:
- **`grid.png`**: Imagen de cuadrícula para visualización
- **`poke_map.gif`**: GIF animado del mapa de Pokémon
- **`youtube.jpg`**: Miniatura del video de YouTube
- **`sblogo.png`**: Logo de Stable Baselines
- **`pyboy.svg`**: Logo de PyBoy

## Archivos Clave para Comportamiento del Agente

### 1. **Políticas y Acciones del Agente**

**`red_gym_env.py`** (líneas clave):
- **Espacio de acciones**: Define qué botones puede presionar el agente
- **Función de recompensa**: Determina cómo se evalúa el comportamiento
- **Observaciones**: Qué información recibe el agente del juego
- **Exploración**: Sistema de recompensas por explorar nuevas áreas

**`memory_addresses.py`**:
- **Posición del jugador**: `X_POS_ADDRESS`, `Y_POS_ADDRESS`
- **Información del party**: `PARTY_SIZE_ADDRESS`, `LEVELS_ADDRESSES`
- **Estado del juego**: `BADGE_COUNT_ADDRESS`, `MONEY_ADDRESS_*`
- **Eventos**: `EVENT_FLAGS_START_ADDRESS`, `EVENT_FLAGS_END_ADDRESS`

### 2. **Configuración de Estados Iniciales**

**Estados disponibles** (archivos `.state`):
- Modifica `init_state` en la configuración del entorno para cambiar dónde inicia el agente
- Cada estado representa un punto diferente en el progreso del juego

**En `run_pretrained_interactive.py`**:
```python
env_config = {
    'init_state': '../has_pokedex_nballs.state',  # CAMBIAR AQUÍ el estado inicial
    'action_freq': 24,                            # Frecuencia de acciones
    'headless': False,                            # Mostrar ventana del juego
    'max_steps': ep_length,                       # Máximo de pasos por episodio
    # ... más configuraciones
}
```

### 3. **Parámetros Editables del Comportamiento**

**En `red_gym_env.py`** (configuraciones importantes):
- **`explore_weight`**: Peso de la recompensa por exploración
- **`reward_scale`**: Escala general de recompensas
- **`action_freq`**: Frecuencia de ejecución de acciones
- **`similar_frame_dist`**: Distancia para considerar frames similares
- **`use_screen_explore`**: Usar exploración basada en pantalla

**En `memory_addresses.py`**:
- Puedes agregar nuevas direcciones de memoria para acceder a más datos del juego
- Útil para crear nuevas recompensas o condiciones

## Entrenamiento del Modelo

### Versión V2 (Recomendada)
```bash
cd v2
python baseline_fast_v2.py
```

### Versión Original
```bash
cd baselines
python run_baseline_parallel_fast.py
```

## Monitoreo del Progreso

### TensorBoard (Local)
```bash
cd [directorio_de_sesión]
tensorboard --logdir .
```
Luego navegar a `localhost:6006`

### Transmisión en Vivo
El proyecto incluye capacidad para transmitir sesiones de entrenamiento a un mapa global compartido usando `stream_agent_wrapper.py`.

### Visualización Estática
Usar los notebooks en el directorio `visualization/` para análisis detallado del comportamiento del agente.

## Uso Interactivo

Una vez ejecutando `run_pretrained_interactive.py`:
- **Teclas de flecha**: Movimiento
- **A y S**: Botones A y B del Game Boy
- **Pausar IA**: Editar `agent_enabled.txt` (cambiar a `False`)

## Personalización Avanzada

### Modificar Recompensas
Editar la función `_calculate_reward()` en `red_gym_env.py` para cambiar cómo el agente es recompensado.

### Cambiar Estado Inicial
Modificar `init_state` en la configuración del entorno para comenzar desde diferentes puntos del juego.

### Agregar Nuevas Observaciones
Añadir direcciones de memoria en `memory_addresses.py` y modificar `_get_obs()` en el entorno.

### Configurar Exploración
Ajustar `explore_weight` y `use_screen_explore` para cambiar el comportamiento exploratorio.

## Notas Importantes

- El archivo `PokemonRed.gb` DEBE estar en el directorio principal
- El directorio actual DEBE ser `baselines/` o `v2/` al ejecutar scripts
- Python 3.10+ es altamente recomendado para compatibilidad
- Para GPUs AMD, seguir la guía de instalación de PyTorch con ROCm

## Recursos Adicionales

- [Video explicativo en YouTube](https://youtu.be/DcYLT37ImBY)
- [Servidor Discord del proyecto](http://discord.gg/RvadteZk4G)
- [Visualización en vivo del mapa](https://pwhiddy.github.io/pokerl-map-viz/)
- [Repositorio original](https://github.com/PWhiddy/PokemonRedExperiments)

## Solución de Problemas

1. **Error de ROM**: Verificar que `PokemonRed.gb` esté en el directorio correcto con el hash correcto
2. **Problemas de dependencias**: Usar entorno virtual y versión específica de Python
3. **Errores de SDL**: Instalar bibliotecas SDL por separado si es necesario
4. **Rendimiento lento**: Considerar usar la versión V2 y ajustar `action_freq`

---

*Este proyecto es una implementación educativa de aprendizaje por refuerzo aplicado a videojuegos retro. Requiere una copia legal de Pokémon Red.*
