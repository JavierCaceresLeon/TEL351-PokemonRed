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
   cd baselines  # o v2 para la versión mejorada
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
├── analyze_session.py         # Script para analizar sesiones de juego
├── run_controlled_session.py  # Script para sesiones con guardado automático
├── *.state                    # Estados guardados del juego
└── assets/                    # Recursos gráficos y multimedia
```

#### Estados del Juego (*.state)
- **`init.state`**: Estado inicial básico del juego
- **`has_pokedex.state`**: Estado donde el jugador ya tiene la Pokédex
- **`has_pokedex_nballs.state`**: Estado con Pokédex y Pokéballs
- **`fast_text_start.state`**: Estado optimizado para texto rápido

### Directorio `baselines/` (Versión Original)

**Archivos principales:**
- **`red_gym_env.py`**: ARCHIVO CLAVE - Define el entorno de gimnasio principal donde el agente interactúa con el juego
- **`run_pretrained_interactive.py`**: EJECUTAR MODELO - Script para ejecutar el modelo preentrenado de forma interactiva
- **`memory_addresses.py`**: CONFIGURACIÓN DEL JUEGO - Define las direcciones de memoria para acceder a datos del juego
- **`requirements.txt`**: Dependencias de Python requeridas

### Directorio `v2/` (Versión Mejorada - Recomendada)

**Mejoras de la V2:**
- Entrenamiento más rápido y eficiente en memoria
- Sistema de recompensas basado en coordenadas
- Mejor manejo de menús y estados bloqueados
- Streaming al mapa habilitado por defecto

**Archivos principales:**
- **`red_gym_env_v2.py`**: ENTORNO MEJORADO - Versión optimizada del entorno de gimnasio
- **`baseline_fast_v2.py`**: ENTRENAMIENTO V2 - Script principal de entrenamiento de la versión 2
- **`run_pretrained_interactive.py`**: Ejecutor del modelo preentrenado para V2

## Análisis Técnico Detallado para Agentes Inteligentes

### Paradigma de Aprendizaje por Refuerzo vs Búsqueda Tradicional

Este proyecto representa una evolución desde algoritmos de búsqueda tradicionales (como Tabú Search, Greedy, A*) hacia técnicas de aprendizaje por refuerzo profundo. A diferencia de los métodos de búsqueda que requieren conocimiento explícito del espacio de estados y funciones heurísticas, el agente aprende implícitamente las estrategias óptimas a través de la interacción con el entorno.

#### Comparación con Métodos de Búsqueda Tradicionales:

**Búsqueda Tradicional (Tabú, Greedy, etc.):**
- Requiere modelado explícito del espacio de estados
- Función heurística definida manualmente
- Exploración determinística o semi-determinística
- Conocimiento del dominio incorporado por el programador
- Escalabilidad limitada en espacios de estados grandes

**Aprendizaje por Refuerzo (PPO):**
- Aprendizaje automático de políticas óptimas
- Exploración estocástica con balance automático exploración/explotación
- Adaptación a entornos complejos y parcialmente observables
- Generalización a estados no vistos durante entrenamiento
- Escalabilidad mejorada mediante aproximación de funciones

### Arquitectura del Agente Inteligente

#### Algoritmo Principal: PPO (Proximal Policy Optimization)

PPO es un algoritmo de Policy Gradient que resuelve el problema de estabilidad en el entrenamiento de políticas. Sus características principales:

**Función Objetivo PPO:**
```
L^CLIP(θ) = E[min(rt(θ)At, clip(rt(θ), 1-ε, 1+ε)At)]
```
Donde:
- rt(θ) = π_θ(at|st) / π_θ_old(at|st) (ratio de probabilidades)
- At = ventaja estimada en el tiempo t
- ε = parámetro de clipping (típicamente 0.2)

**Ventajas de PPO en este dominio:**
1. **Estabilidad**: Evita cambios drásticos en la política
2. **Eficiencia de muestra**: Reutiliza datos de episodios anteriores
3. **Paralelización**: Entrenamiento en múltiples entornos simultáneos (64 procesos)
4. **Robustez**: Manejo efectivo de espacios de acción discretos

#### Espacio de Estados y Observaciones

El agente recibe observaciones multi-modales que incluyen:

```python
observation_space = spaces.Dict({
    "screens": Box(0, 255, (72, 80, frame_stacks), uint8),     # Frames del juego
    "health": Box(0, 1),                                        # Salud normalizada  
    "level": Box(-1, 1, (enc_freqs,)),                        # Niveles codificados
    "badges": MultiBinary(8),                                   # Vector de medallas
    "events": MultiBinary((event_flags_end - event_flags_start) * 8), # Flags de eventos
    "map": Box(0, 255, (coords_pad*4, coords_pad*4, 1)),      # Mapa de exploración
    "recent_actions": MultiDiscrete([n_actions] * frame_stacks) # Historial de acciones
})
```

**Técnicas de Representación Avanzadas:**

1. **Codificación Fourier para Niveles:**
```python
def fourier_encode(self, value):
    # Mapea valores continuos a representación sinusoidal
    # Permite mejor generalización en espacios continuos
    return np.array([np.sin(2 * np.pi * value * freq) for freq in self.freqs])
```

2. **Downsampling de Imágenes:**
- Reducción de 144x160 a 72x80 pixels
- Preserva información espacial crítica
- Reduce dimensionalidad computacional

3. **Stack de Frames:**
- Mantiene historial temporal de 3 frames
- Permite detección de movimiento y dinámicas temporales

#### Espacio de Acciones

```python
valid_actions = [
    PRESS_ARROW_DOWN, PRESS_ARROW_LEFT, PRESS_ARROW_RIGHT, PRESS_ARROW_UP,
    PRESS_BUTTON_A, PRESS_BUTTON_B, PRESS_BUTTON_START
]
```

**Estrategia de Ejecución de Acciones:**
- Presión de botón por 8 ticks
- Liberación automática
- Continuación por remaining ticks (24 total por acción)
- Previene acciones "pegadas" que causaban problemas en V1

### Sistema de Recompensas: Ingeniería de Recompensas Sofisticada

El sistema de recompensas en V2 representa una evolución significativa desde sistemas heurísticos simples:

#### Componentes de Recompensa Detallados:

```python
state_scores = {
    "event": reward_scale * update_max_event_rew() * 4,        # Eventos del juego
    "heal": reward_scale * total_healing_rew * 10,             # Curación
    "badge": reward_scale * get_badges() * 10,                 # Medallas de gimnasio  
    "explore": reward_scale * explore_weight * len(seen_coords) * 0.1, # Exploración
    "stuck": reward_scale * get_current_coord_count_reward() * -0.05   # Anti-bucle
}
```

**1. Recompensa de Exploración (Clave de V2):**
- **V1**: Comparación de frames con k-NN (computacionalmente costoso)
- **V2**: Basada en coordenadas únicas visitadas
- **Ventaja**: Evita redundancia visual, enfoca en diversidad espacial
- **Implementación**: `len(seen_coords)` - conteo de ubicaciones (x,y,map) únicas

**2. Recompensa de Eventos:**
- Monitoreo de 319 flags de eventos del juego (0xD747 a 0xD886)
- Cada bit representa un evento específico (obtener Pokédex, hablar con NPCs, etc.)
- Recompensa acumulativa que incentiva progreso narrativo

**3. Sistema Anti-Bloqueo:**
- **Problema en V1**: Agente se quedaba en menús o bucles
- **Solución V2**: Recompensa negativa por estar demasiado tiempo en misma coordenada
- **Implementación**: Penalización progresiva basada en frecuencia de visita

**4. Recompensa de Curación:**
- Incentiva mantener Pokémon con vida
- Bonus cuadrático: `heal_amount * heal_amount`
- Penalización por muerte: `died_count` incrementa

**5. Medallas como Objetivos de Alto Nivel:**
- Recompensa máxima (10x) por obtener medallas de gimnasio
- Representa objetivos a largo plazo en el juego

### Estrategias de Entrenamiento Avanzadas

#### Paralelización Masiva

```python
num_cpu = 64  # 64 entornos paralelos
env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
```

**Ventajas:**
- **Diversidad de Experiencia**: 64 trayectorias simultáneas
- **Estabilidad de Gradientes**: Promediado sobre múltiples entornos
- **Eficiencia**: Paralelización a nivel de CPU
- **Exploración**: Diferentes semillas iniciales por proceso

#### Configuración PPO Optimizada

```python
model = PPO("MultiInputPolicy", env,
    n_steps=2048,        # Pasos por actualización de política
    batch_size=512,      # Tamaño de lote para optimización
    n_epochs=1,          # Épocas por actualización (previene overfitting)
    gamma=0.997,         # Factor de descuento (horizonte largo)
    ent_coef=0.01,       # Coeficiente de entropía (exploración)
    learning_rate=3e-4   # Tasa de aprendizaje
)
```

**Justificaciones Técnicas:**
- **n_steps=2048**: Balance entre varianza y bias en estimación de gradientes
- **n_epochs=1**: Previene overfitting en datos de episodios
- **gamma=0.997**: Horizonte temporal largo (recompensas a 300+ pasos)
- **ent_coef=0.01**: Mantiene exploración sin sacrificar convergencia

### Manejo del Problema de Menús (V1 vs V2)

#### Problema Original (V1):
- **Síntoma**: Agente se quedaba en menú principal
- **Causa**: Falta de diversidad en estados de menú durante entrenamiento
- **Enfoque**: Penalizaciones manuales (solución parcial)

#### Solución Elegante (V2):
- **Exploración Basada en Coordenadas**: Fuerza diversidad espacial
- **Anti-Stuck Reward**: Penalización automática por permanencia
- **Mejor Representación de Estados**: Incluye contexto de acciones recientes
- **Estados Iniciales Diversos**: Entrenamiento desde múltiples puntos

### Innovaciones Potenciales para Extensión

#### 1. Búsqueda Híbrida RL + Tradicional
```python
class HybridAgent:
    def __init__(self, rl_policy, search_planner):
        self.rl_policy = rl_policy
        self.search_planner = search_planner
    
    def select_action(self, state):
        # RL para exploración general
        rl_action = self.rl_policy.predict(state)
        
        # Búsqueda para objetivos específicos
        if self.in_specific_context(state):
            return self.search_planner.plan(state, objective)
        return rl_action
```

#### 2. Curriculum Learning
- Entrenamiento progresivo desde objetivos simples a complejos
- Inicio en Pallet Town → Viridian City → ... → Elite Four

#### 3. Hierarchical Reinforcement Learning
- Políticas de alto nivel (objetivos) y bajo nivel (movimientos)
- Descomposición de tarea en sub-objetivos

#### 4. Meta-Learning
- Adaptación rápida a nuevas versiones de Pokémon
- Transfer learning entre diferentes juegos de la serie

## Uso de analyze_session.py

### Sintaxis Básica

```bash
# Analizar una sesión específica
python analyze_session.py v2/session_752558fa

# Si ya estás en el directorio v2/
cd v2
python ../analyze_session.py session_752558fa
```

### Información Proporcionada

El script muestra:

1. **Estadísticas Básicas:**
   - Total de pasos ejecutados
   - Duración estimada en minutos de juego
   - Ubicaciones únicas visitadas
   - Nivel máximo alcanzado
   - Medallas obtenidas
   - Número de muertes

2. **Análisis de Mapas:**
   - Lista de mapas visitados con nombres
   - Tiempo gastado en cada ubicación
   - Patrones de movimiento

3. **Gráficos Automáticos:**
   - Progreso de exploración vs tiempo
   - Evolución de niveles de Pokémon
   - Fluctuaciones de salud del party
   - Guardado automático como PNG

4. **Archivos de Datos:**
   - Conteo de screenshots disponibles
   - Estados finales capturados
   - Archivos JSON con resúmenes

### Interpretación de Resultados

```bash
# Ejemplo de salida
Analizando sesión: session_752558fa
==================================================
Estadísticas Básicas:
  • Total de pasos: 8,459
  • Duración: ~141.0 minutos de juego  
  • Ubicaciones únicas: 342
  • Nivel máximo alcanzado: 15
  • Medallas obtenidas: 1
  • Muertes: 2

Mapas visitados (15):
  • Pallet Town: 1,203 pasos
  • Route 1: 856 pasos
  • Viridian City: 2,341 pasos
  ...

Gráfico guardado en: session_752558fa/analysis_plot.png
Screenshots disponibles: 169
Estados finales: 3
```

## Guardar Sesiones sin Interrupción Manual

### Problema con Ctrl+C

Cuando usas Ctrl+C para terminar una sesión interactiva:
- El proceso se interrumpe abruptamente
- No se ejecutan las rutinas de guardado
- Se pierden estadísticas y datos de la sesión
- Los directorios de sesión quedan vacíos

### Solución 1: Script de Sesión Controlada

Usa el script `run_controlled_session.py` incluido:

```bash
# Ejecutar sesión de 10,000 pasos con guardado cada 500 pasos
python run_controlled_session.py 10000 500

# Especificar modelo personalizado
python run_controlled_session.py 5000 250 v2/runs/poke_26214400

# Parámetros: [max_steps] [save_frequency] [checkpoint_path]
```

**Ventajas del script controlado:**
- Guardado automático periódico
- Manejo correcto de Ctrl+C (guardado antes de terminar)
- Reportes de progreso en tiempo real
- Preservación completa de datos de sesión

### Solución 2: Configurar Límites de Tiempo

```python
# Modificar parámetros en run_pretrained_interactive.py
env_config = {
    'max_steps': 5000,          # Episodios más cortos
    'save_final_state': True,   # Forzar guardado al terminar
    'print_rewards': True,      # Ver progreso en consola
}
```

### Solución 3: Monitoreo con TensorBoard

```bash
# Terminal 1: Ejecutar sesión/entrenamiento
cd v2
python baseline_fast_v2.py  # o run_pretrained_interactive.py

# Terminal 2: Monitorear progreso en tiempo real
cd v2/runs  # o directorio de sesión correspondiente
tensorboard --logdir .

# Abrir navegador en localhost:6006
```

## Archivos Clave para Comportamiento del Agente

### 1. Políticas y Acciones del Agente

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

### 2. Configuración de Estados Iniciales

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

## Uso Interactivo

Una vez ejecutando `run_pretrained_interactive.py`:
- **Teclas de flecha**: Movimiento
- **A y S**: Botones A y B del Game Boy
- **Pausar IA**: Editar `agent_enabled.txt` (cambiar a `False`)

## Personalización Avanzada

### Modificar Recompensas
Editar las funciones de recompensa en `red_gym_env_v2.py` para cambiar el comportamiento del agente.

### Cambiar Estado Inicial
Modificar `init_state` en la configuración del entorno para comenzar desde diferentes puntos del juego.

### Agregar Nuevas Observaciones
Añadir direcciones de memoria en `memory_addresses.py` y modificar las observaciones del entorno.

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
5. **Error PyBoy V2 - APIs obsoletas**: 
   - **Problema 1 - Memoria**: `AttributeError: 'PyBoy' object has no attribute 'memory'`
     ```python
     # En v2/red_gym_env_v2.py, cambiar:
     return self.pyboy.memory[addr]
     # A:
     return self.pyboy.get_memory_value(addr)
     ```
   - **Problema 2 - Pantalla**: `AttributeError: 'PyBoy' object has no attribute 'screen'`
     ```python
     # Descomentar y corregir en v2/red_gym_env_v2.py:
     self.screen = self.pyboy.botsupport_manager().screen()
     # Y cambiar:
     game_pixels_render = self.pyboy.screen.ndarray[:,:,0:1]
     # A:
     game_pixels_render = self.screen.screen_ndarray()[:,:,0:1]
     ```
   - **Problema 3 - Tick**: `TypeError: tick() takes exactly 0 positional arguments`
     ```python
     # Cambiar de:
     self.pyboy.tick(press_step, render_screen)
     # A:
     self.pyboy.tick()
     ```
   - **Nota**: Estos errores ya están corregidos en este repositorio
6. **Error "Could not deserialize object tensorboard_log"**: 
   - Es una advertencia, no afecta la ejecución
   - Relacionado con compatibilidad de rutas entre Windows y sistemas Unix

---

*Este proyecto es una implementación educativa de aprendizaje por refuerzo aplicado a videojuegos retro. Requiere una copia legal de Pokémon Red.*

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
   cd baselines  # o v2 para la versión mejorada
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
├── analyze_session.py         # Script para analizar sesiones de juego
├── run_controlled_session.py  # Script para sesiones con guardado automático
├── *.state                    # Estados guardados del juego
└── assets/                    # Recursos gráficos y multimedia
```

#### Estados del Juego (*.state)
- **`init.state`**: Estado inicial básico del juego
- **`has_pokedex.state`**: Estado donde el jugador ya tiene la Pokédex
- **`has_pokedex_nballs.state`**: Estado con Pokédex y Pokéballs
- **`fast_text_start.state`**: Estado optimizado para texto rápido

### Directorio `baselines/` (Versión Original)

**Archivos principales:**
- **`red_gym_env.py`**: Entorno de gimnasio principal donde el agente interactúa
- **`run_pretrained_interactive.py`**: Script para ejecutar el modelo preentrenado
- **`memory_addresses.py`**: Direcciones de memoria para acceder a datos del juego
- **`requirements.txt`**: Dependencias de Python requeridas

### Directorio `v2/` (Versión Mejorada - Recomendada)

**Mejoras de la V2:**
- Entrenamiento más rápido y eficiente en memoria
- Sistema de recompensas basado en coordenadas
- Mejor manejo de menús y estados bloqueados
- Streaming al mapa habilitado por defecto

**Archivos principales:**
- **`red_gym_env_v2.py`**: Entorno optimizado con mejoras técnicas
- **`baseline_fast_v2.py`**: Script principal de entrenamiento
- **`run_pretrained_interactive.py`**: Ejecutor del modelo preentrenado para V2

## Uso de analyze_session.py

### Sintaxis Básica

```bash
# Analizar una sesión específica
python analyze_session.py v2/session_752558fa

# Si ya estás en el directorio v2/
cd v2
python ../analyze_session.py session_752558fa
```

### Información Proporcionada

El script muestra:

1. **Estadísticas Básicas:**
   - Total de pasos ejecutados
   - Duración estimada en minutos de juego
   - Ubicaciones únicas visitadas
   - Nivel máximo alcanzado
   - Medallas obtenidas
   - Número de muertes

2. **Análisis de Mapas:**
   - Lista de mapas visitados con nombres
   - Tiempo gastado en cada ubicación
   - Patrones de movimiento

3. **Gráficos Automáticos:**
   - Progreso de exploración vs tiempo
   - Evolución de niveles de Pokémon
   - Fluctuaciones de salud del party
   - Guardado automático como PNG

4. **Archivos de Datos:**
   - Conteo de screenshots disponibles
   - Estados finales capturados
   - Archivos JSON con resúmenes

### Interpretación de Resultados

```bash
# Ejemplo de salida
Analizando sesión: session_752558fa
==================================================
Estadísticas Básicas:
  • Total de pasos: 8,459
  • Duración: ~141.0 minutos de juego  
  • Ubicaciones únicas: 342
  • Nivel máximo alcanzado: 15
  • Medallas obtenidas: 1
  • Muertes: 2

Mapas visitados (15):
  • Pallet Town: 1,203 pasos
  • Route 1: 856 pasos
  • Viridian City: 2,341 pasos
  ...

Gráfico guardado en: session_752558fa/analysis_plot.png
Screenshots disponibles: 169
Estados finales: 3
```

## Guardar Sesiones sin Interrupción Manual

### Problema con Ctrl+C

Cuando usas Ctrl+C para terminar una sesión interactiva:
- El proceso se interrumpe abruptamente
- No se ejecutan las rutinas de guardado
- Se pierden estadísticas y datos de la sesión
- Los directorios de sesión quedan vacíos

### Solución 1: Script de Sesión Controlada

Usa el script `run_controlled_session.py` incluido:

```bash
# Ejecutar sesión de 10,000 pasos con guardado cada 500 pasos
python run_controlled_session.py 10000 500

# Especificar modelo personalizado
python run_controlled_session.py 5000 250 v2/runs/poke_26214400

# Parámetros: [max_steps] [save_frequency] [checkpoint_path]
```

**Ventajas del script controlado:**
- Guardado automático periódico
- Manejo correcto de Ctrl+C (guardado antes de terminar)
- Reportes de progreso en tiempo real
- Preservación completa de datos de sesión

### Solución 2: Configurar Límites de Tiempo

```python
# Modificar parámetros en run_pretrained_interactive.py
env_config = {
    'max_steps': 5000,          # Episodios más cortos
    'save_final_state': True,   # Forzar guardado al terminar
    'print_rewards': True,      # Ver progreso en consola
}
```

### Solución 3: Monitoreo con TensorBoard

```bash
# Terminal 1: Ejecutar sesión/entrenamiento
cd v2
python baseline_fast_v2.py  # o run_pretrained_interactive.py

# Terminal 2: Monitorear progreso en tiempo real
cd v2/runs  # o directorio de sesión correspondiente
tensorboard --logdir .

# Abrir navegador en localhost:6006
```

Esto permite:
- Ver métricas en tiempo real
- Decidir cuándo terminar basado en progreso
- Obtener datos incluso si se interrumpe el proceso

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

## Uso Interactivo

Una vez ejecutando `run_pretrained_interactive.py`:
- **Teclas de flecha**: Movimiento
- **A y S**: Botones A y B del Game Boy
- **Pausar IA**: Editar `agent_enabled.txt` (cambiar a `False`)

## Personalización Avanzada

### Modificar Recompensas
Editar las funciones de recompensa en `red_gym_env_v2.py` para cambiar el comportamiento del agente.

### Cambiar Estado Inicial
Modificar `init_state` en la configuración del entorno para comenzar desde diferentes puntos del juego.

### Agregar Nuevas Observaciones
Añadir direcciones de memoria en `memory_addresses.py` y modificar las observaciones del entorno.

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
5. **Error PyBoy V2 - APIs obsoletas**: 
   - **Problema 1 - Memoria**: `AttributeError: 'PyBoy' object has no attribute 'memory'`
     ```python
     # En v2/red_gym_env_v2.py, cambiar:
     return self.pyboy.memory[addr]
     # A:
     return self.pyboy.get_memory_value(addr)
     ```
   - **Problema 2 - Pantalla**: `AttributeError: 'PyBoy' object has no attribute 'screen'`
     ```python
     # Descomentar y corregir en v2/red_gym_env_v2.py:
     self.screen = self.pyboy.botsupport_manager().screen()
     # Y cambiar:
     game_pixels_render = self.pyboy.screen.ndarray[:,:,0:1]
     # A:
     game_pixels_render = self.screen.screen_ndarray()[:,:,0:1]
     ```
   - **Problema 3 - Tick**: `TypeError: tick() takes exactly 0 positional arguments`
     ```python
     # Cambiar de:
     self.pyboy.tick(press_step, render_screen)
     # A:
     self.pyboy.tick()
     ```
   - **Nota**: Estos errores ya están corregidos en este repositorio
6. **Error "Could not deserialize object tensorboard_log"**: 
   - Es una advertencia, no afecta la ejecución
   - Relacionado con compatibilidad de rutas entre Windows y sistemas Unix

---

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
