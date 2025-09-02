# Entrenamiento de Agentes de Aprendizaje por Refuerzo para Pokémon Red

Este proyecto### 🔧 Script de Análisis Automático - analyze_session.py

#### 📋 Uso Detallado del Script

**Sintaxis:**
```bash
python analyze_session.py [directorio_de_sesión]
```

**Ejemplos Prácticos:**
```bash
# Desde el directorio raíz del proyecto
python analyze_session.py v2/session_752558fa

# Desde dentro del directorio v2/
cd v2
python ../analyze_session.py session_752558fa

# Analizar la sesión más reciente (encontrar automáticamente)
python ../analyze_session.py $(ls -td session_*/ | head -1)
```

**Lo que hace el script paso a paso:**

1. **Verificación de archivos**: Busca `agent_stats_*.csv.gz` en el directorio
2. **Carga de datos**: Lee el archivo CSV comprimido con pandas
3. **Análisis estadístico**: Calcula métricas de rendimiento
4. **Visualización**: Genera gráficos de progreso automáticamente
5. **Reporte**: Muestra resumen en consola

**Salida del script:**
```
🔍 Analizando sesión: session_752558fa
==================================================
📊 Cargando: agent_stats_abc123.csv.gz

📈 Estadísticas Básicas:
  • Total de pasos: 15,432
  • Duración: ~257.2 minutos de juego
  • Ubicaciones únicas: 1,234
  • Nivel máximo alcanzado: 12
  • Medallas obtenidas: 1
  • Muertes: 3

🗺️ Mapas visitados (8):
  • Pallet Town: 5,432 pasos
  • Route 1: 3,221 pasos
  • Viridian City: 2,100 pasos
  ...

📊 Gráfico guardado en: session_752558fa/analysis_plot.png
🏆 Resumen de ejecuciones (1 runs):
  • event: 245.67
  • level: 89.32
  • explore: 156.78
  ...
📸 Screenshots disponibles: 308
🎯 Estados finales: 2
```

#### 💾 Cómo Guardar Sesiones Completas (Sin Ctrl+C)

**Problema**: Ctrl+C termina abruptamente y no guarda datos.

**Soluciones:**

**Método 1: Usar el archivo agent_enabled.txt**
```bash
# 1. Iniciar el agente
python run_pretrained_interactive.py

# 2. En otra terminal, pausar el agente
echo "False" > agent_enabled.txt

# 3. Para reanudar
echo "True" > agent_enabled.txt

# 4. Para terminar limpiamente
echo "False" > agent_enabled.txt
# Luego presionar 'q' en la ventana del juego
```

**Método 2: Modificar max_steps en run_pretrained_interactive.py**
```python
# Línea ~33 en run_pretrained_interactive.py
env_config = {
    'max_steps': 5000,  # Cambiar de 2**23 a un número menor
    # ... resto de configuración
}
```

**Método 3: Usar timeout en la terminal**
```bash
# Ejecutar por máximo 10 minutos
timeout 600 python run_pretrained_interactive.py
```

**Método 4: Script de ejecución automática** (recomendado para análisis)
```bash
# Usar el script incluido para sesiones controladas
python run_controlled_session.py 5        # 5 minutos con ventana
python run_controlled_session.py 10 True  # 10 minutos sin ventana (headless)

# El script automáticamente:
# 1. Ejecuta el agente por el tiempo especificado
# 2. Termina limpiamente el proceso
# 3. Encuentra la sesión generada
# 4. Ejecuta el análisis automáticamente
```

**Método 5: Control manual con agent_enabled.txt**
```python
# Crear archivo: run_analysis_session.py
import subprocess
import time
import signal
import os

def run_timed_session(duration_minutes=10):
    process = subprocess.Popen(['python', 'run_pretrained_interactive.py'])
    time.sleep(duration_minutes * 60)
    process.send_signal(signal.SIGTERM)  # Terminación limpia
    process.wait()

if __name__ == "__main__":
    run_timed_session(5)  # 5 minutos de sesión
```no de aprendizaje por refuerzo para entrenar agentes de IA que jueguen Pokémon Red automáticamente. El agente aprende a nave## 📊 Monitoreo del Progreso y Revisión de Sesiones

### 🎮 Sesiones de Juego Interactivas

Cuando ejecutas `run_pretrained_interactive.py`, se generan varios tipos de archivos:

#### Directorios de Sesión
- **Ubicación**: `v2/session_[ID_ÚNICO]/` (ej: `session_752558fa/`)
- **Contenido**: Screenshots, estadísticas, estados finales
- **Cuándo se crean**: Solo cuando el agente completa una sesión completa o llega al límite de pasos

#### Archivos Generados por Sesión:
```
session_[ID]/
├── curframe_[ID].jpeg          # Screenshot del frame actual (cada 50 pasos)
├── agent_stats_[ID].csv.gz     # Estadísticas detalladas del agente
├── all_runs_[ID].json          # Resumen de todas las ejecuciones
└── final_states/               # Estados finales cuando termina
    ├── frame_r[score]_full.jpeg
    └── frame_r[score]_small.jpeg
```

#### ⭐ Archivos Clave para Analizar el Comportamiento:

**1. `agent_stats_[ID].csv.gz`** - Contiene datos paso a paso:
- Posición (x, y) del jugador
- Mapa actual
- Última acción tomada
- Niveles de Pokémon
- Puntos de vida (HP)
- Exploración realizada
- Medallas obtenidas
- Eventos completados

**2. `curframe_[ID].jpeg`** - Screenshots automáticos cada 50 pasos

**3. `all_runs_[ID].json`** - Puntuaciones de recompensas

## � Análisis Técnico Detallado del Entrenamiento (Para Agentes Inteligentes)

### 🔬 Diferencia Fundamental: Búsqueda vs Aprendizaje por Refuerzo

#### Algoritmos de Búsqueda Clásicos (Lo que NO usa este proyecto):
- **Tabú Search**: Mantiene lista de movimientos prohibidos para evitar ciclos
- **Greedy**: Toma decisiones basadas en el mejor resultado inmediato
- **A***: Busca camino óptimo con heurística
- **Limitación**: Requieren conocimiento previo del espacio de estados y función objetivo

#### Aprendizaje por Refuerzo (Lo que SÍ usa):
- **No requiere conocimiento previo** del juego
- **Aprende mediante interacción** con el entorno
- **Optimiza recompensas a largo plazo**, no solo inmediatas
- **Generaliza** a situaciones no vistas durante entrenamiento

### 🎯 Sistema de Recompensas/Penalizaciones Detallado

#### 📍 Ubicación del Código de Recompensas:
- **Archivo principal**: `v2/red_gym_env_v2.py`
- **Función clave**: `get_game_state_reward()` (línea ~400)
- **Configuración**: `baseline_fast_v2.py` (parámetros de entrenamiento)

#### 🏆 Tipos de Recompensas (red_gym_env_v2.py):

**1. Recompensas de Exploración:**
```python
# Línea ~380 en red_gym_env_v2.py
def get_knn_reward(self):
    # Coordenadas únicas visitadas
    cur_size = len(self.seen_coords)
    base = self.base_explore * 0.005  # Exploración base
    post = cur_size * 0.01 if self.levels_satisfied else 0
    return base + post
```
- **Propósito**: Incentiva explorar nuevas áreas del mapa
- **Implementación**: +0.005 puntos por coordenada nueva antes de nivel 22
- **Bonus**: +0.01 puntos después del nivel 22 (exploración avanzada)

**2. Recompensas de Progreso de Niveles:**
```python
# Línea ~350
def get_levels_reward(self):
    level_sum = self.get_levels_sum()
    if level_sum < 22:  # Exploración temprana
        scaled = level_sum
    else:  # Progreso avanzado
        scaled = (level_sum-22) / 4 + 22
    return max(self.max_level_rew, scaled)
```
- **Propósito**: Recompensa por subir niveles de Pokémon
- **Escalamiento**: Lineal hasta nivel 22, luego escalado para evitar sobreajuste

**3. Recompensas por Eventos del Juego:**
```python
# Línea ~420
def get_all_events_reward(self):
    # Suma flags de eventos (obtener Pokédex, items, etc.)
    base_event_flags = 13  # Estados iniciales
    total_events = sum([self.bit_count(self.read_m(i)) 
                       for i in range(event_flags_start, event_flags_end)])
    return max(total_events - base_event_flags, 0)
```
- **Propósito**: Recompensa objetivos específicos del juego
- **Ejemplos**: Obtener Pokédex (+puntos), conseguir items (+puntos)

**4. Recompensas por Medallas:**
```python
# Línea ~440 en get_game_state_reward()
'badge': self.reward_scale * self.get_badges() * 5
```
- **Propósito**: Gran bonificación por derrotar líderes de gimnasio
- **Multiplicador**: x5 la escala base de recompensas

**5. Sistema de Salud y Penalizaciones:**
```python
# Línea ~310
def update_heal_reward(self):
    cur_health = self.read_hp_fraction()
    if cur_health > self.last_health:
        heal_amount = cur_health - self.last_health
        self.total_healing_rew += heal_amount * 4  # +4 por curación
    else:
        self.died_count += 1  # Contador de muertes

# En get_game_state_reward():
'heal': self.reward_scale * self.total_healing_rew,
'dead': self.reward_scale * -0.1 * self.died_count,  # -0.1 por muerte
```

#### 🚫 Por Qué V2 No Se Queda en Menús (Solución al Problema de V1)

**Problema en V1**: El agente se quedaba atascado en menús sin recompensas claras.

**Soluciones en V2:**

**1. Observaciones Enriquecidas:**
```python
# V2 usa observaciones estructuradas (línea ~185)
observation_space = spaces.Dict({
    "screens": spaces.Box(...),      # Pantalla del juego
    "health": spaces.Box(...),       # Estado de salud
    "level": spaces.Box(...),        # Niveles codificados
    "badges": spaces.MultiBinary(8), # Medallas obtenidas
    "events": spaces.MultiBinary(...), # Eventos del juego
    "map": spaces.Box(...),          # Posición en mapa
    "recent_actions": spaces.MultiDiscrete(...) # Historial de acciones
})
```

**2. Penalizaciones Implícitas por Inactividad:**
```python
# Si no hay progreso en coordenadas nuevas = sin recompensa de exploración
# Si no hay progreso en niveles = sin recompensa de niveles
# Resultado: El agente aprende que estar inmóvil = sin recompensas
```

**3. Recompensas por Exploración Continua:**
```python
# update_seen_coords() premia movimiento constante
self.seen_coords[coord_string] = self.step_count
# Solo obtiene recompensas si visita coordenadas nuevas
```

### 🤖 Algoritmo de Entrenamiento: PPO (Proximal Policy Optimization)

#### 📚 Fundamentos Teóricos:

**¿Qué es PPO?**
- **Familia**: Actor-Critic (combina Policy Gradient + Value Function)
- **Innovación**: Actualiza política de forma conservadora para evitar degradación
- **Ventaja**: Más estable que TRPO, más eficiente que A2C

**Arquitectura del Agente:**
```python
# baseline_fast_v2.py línea ~85
model = PPO("MultiInputPolicy", env, 
    n_steps=2048,           # Pasos por actualización de política
    batch_size=512,         # Tamaño de lote para entrenamiento
    n_epochs=1,             # Épocas por actualización
    gamma=0.997,            # Factor de descuento (0.997 = considera futuro lejano)
    ent_coef=0.01,          # Coeficiente de entropía (fomenta exploración)
)
```

#### 🔄 Ciclo de Entrenamiento Detallado:

**Fase 1: Recolección de Experiencias**
```python
# 64 entornos en paralelo × 2048 pasos = 131,072 experiencias por iteración
num_cpu = 64
n_steps = 2048
```

1. **64 agentes independientes** juegan simultáneamente
2. Cada uno recolecta **2048 pasos** de experiencia
3. Total: **131,072 transiciones** (estado, acción, recompensa, siguiente_estado)

**Fase 2: Cálculo de Ventajas**
```python
# Temporal Difference Learning con γ=0.997
advantage = reward + γ * V(next_state) - V(current_state)
```

**Fase 3: Actualización de Política**
```python
# Función objetivo de PPO
L_PPO = min(
    ratio * advantage,  # ratio = π_new(a|s) / π_old(a|s)
    clip(ratio, 1-ε, 1+ε) * advantage  # ε ≈ 0.2
)
```

**Fase 4: Repetición**
- **Total timesteps**: 26,214,400 (número en el archivo del modelo)
- **Iteraciones**: ~200 (26M ÷ 131K por iteración)
- **Tiempo real**: ~18-24 horas en hardware moderno

### 🏗️ Arquitectura de Red Neural

#### 🧠 Estructura del Modelo (MultiInputPolicy):

**Encoder de Imágenes (CNN):**
```python
# Procesa pantallas del juego (72×80×3)
Conv2D(32, 8×8, stride=4) → ReLU
Conv2D(64, 4×4, stride=2) → ReLU  
Conv2D(64, 3×3, stride=1) → ReLU
Flatten() → Dense(512)
```

**Encoder de Features Categóricas:**
```python
# Procesa badges, events, levels, health
Concatenate(all_features) → Dense(256) → ReLU
```

**Fusion Layer:**
```python
# Combina visual + categorical
Concatenate(image_features, categorical_features) → Dense(512)
```

**Heads de Salida:**
```python
# Actor (política)
Dense(256) → ReLU → Dense(n_actions) → Softmax

# Critic (función de valor)
Dense(256) → ReLU → Dense(1) → Linear
```

### 📊 Métricas de Evaluación del Agente

#### 🎯 KPIs Principales:
1. **Reward Total**: Suma ponderada de todas las recompensas
2. **Episode Length**: Pasos antes de terminar/morir
3. **Exploration Coverage**: Coordenadas únicas visitadas
4. **Level Progress**: Suma de niveles de Pokémon
5. **Event Completion**: Objetivos del juego completados

#### 📈 Evolución del Entrenamiento:
```
Iteración 1-20:   Random exploration, mucho dying
Iteración 21-50:  Aprende movimientos básicos
Iteración 51-100: Optimiza rutas, evita peligros
Iteración 101-150: Estrategias de combate
Iteración 151-200: Comportamiento experto
```

### 🔧 Parámetros Críticos del Entrenamiento

#### ⚙️ Hiperparámetros Clave:
```python
# En baseline_fast_v2.py
env_config = {
    'reward_scale': 0.5,      # Escala global de recompensas
    'explore_weight': 0.25,   # Peso de exploración vs progreso
    'action_freq': 24,        # Frames entre acciones (control de velocidad)
    'max_steps': 163840,      # Límite de pasos por episodio
}
```

#### 🎮 Por Qué Estos Valores Funcionan:
- **reward_scale=0.5**: Evita recompensas excesivamente altas que desestabilicen entrenamiento
- **explore_weight=0.25**: Balance 75% progreso / 25% exploración
- **action_freq=24**: ~2.5 acciones/segundo (velocidad humana razonable)

### 🚀 Innovaciones Técnicas Implementadas

#### 🔬 Técnicas Avanzadas Utilizadas:

**1. Exploration Scheduling:**
```python
# Cambia estrategia de exploración según progreso
if self.get_levels_sum() >= 22:
    self.levels_satisfied = True
    self.base_explore = len(self.seen_coords)
    self.seen_coords = {}  # Reset para nueva fase
```

**2. Multi-Modal Observations:**
- Combina información visual + simbólica
- Permite al agente "entender" el estado del juego más allá de píxeles

**3. Curriculum Learning Implícito:**
- Recompensas escaladas según progreso
- Diferentes objetivos en diferentes fases del juego

**4. Parallel Environment Training:**
- 64 entornos simultáneos aumentan diversidad de experiencias
- Acelera convergencia significativamente

### � Script de Análisis Automático

Hemos incluido un script para analizar fácilmente las sesiones:

```bash
# Analizar una sesión específica
python analyze_session.py v2/session_752558fa

# Si ya estás en el directorio v2/
cd v2
python ../analyze_session.py session_752558fa
```

**El script muestra:**
- Estadísticas básicas (pasos, duración, exploración)
- Mapas visitados y tiempo en cada uno
- Progreso de niveles y salud
- Gráficos de progreso (guardados como PNG)
- Resumen de recompensas
- Conteo de screenshots disponibles

### 🎮 Consejos para Generar Sesiones Completas

Para que se guarden datos completos:
1. **Deja que el agente complete más pasos**: No termines inmediatamente con Ctrl+C
2. **Configura límites apropiados**: El agente guarda datos al completar episodios
3. **Usa modo no interactivo**: Para entrenar y generar datos automáticamente

#### 1. Usar Pandas para Analizar Estadísticas:
```python
import pandas as pd
import gzip

# Cargar estadísticas de una sesión
df = pd.read_csv('session_[ID]/agent_stats_[ID].csv.gz', compression='gzip')

# Ver progreso del agente
print(df[['step', 'x', 'y', 'map', 'levels_sum', 'badge']].head(20))

# Analizar exploración
print(f"Ubicaciones únicas visitadas: {df[['x', 'y', 'map']].drop_duplicates().shape[0]}")

# Ver progreso de niveles
print(df['levels_sum'].plot())
```

#### 2. Examinar Screenshots:
Las imágenes `curframe_*.jpeg` muestran exactamente lo que vio el agente en diferentes momentos.

#### 3. Usar TensorBoard para Métricas Detalladas:
```bash
cd v2/runs
tensorboard --logdir .
```
Navegar a `localhost:6006` para ver:
- Curvas de recompensa
- Pérdidas del modelo
- Métricas de exploración
- Progreso de entrenamiento

### 🎓 Oportunidades de Innovación para Proyecto Universitario

#### � Áreas de Investigación Sugeridas:

**1. Hybrid Search-RL Approaches:**
```python
# Combinar búsqueda clásica con RL
class HybridAgent:
    def __init__(self):
        self.rl_policy = PPO_policy()      # Para exploración general
        self.astar_navigator = AStarAgent() # Para navegación específica
        self.tabu_search = TabuAgent()     # Para evitar loops locales
    
    def select_action(self, state):
        if self.is_navigation_task(state):
            return self.astar_navigator.get_action(state)
        elif self.is_stuck_situation(state):
            return self.tabu_search.get_action(state)
        else:
            return self.rl_policy.get_action(state)
```

**2. Hierarchical Goal Planning:**
```python
# Sistema jerárquico de objetivos
class GoalHierarchy:
    def __init__(self):
        self.long_term_goals = ["get_badge_1", "reach_cerulean"]
        self.short_term_goals = ["find_pokecenter", "buy_pokeballs"]
        self.immediate_actions = ["move_up", "interact", "menu"]
    
    def decompose_goal(self, high_level_goal):
        # Descomponer objetivo en sub-tareas alcanzables
        return self.goal_decomposition[high_level_goal]
```

**3. Multi-Agent Cooperation:**
```python
# Múltiples agentes especializados
class SpecializedAgents:
    def __init__(self):
        self.explorer_agent = ExplorerAgent()    # Especialista en exploración
        self.battle_agent = BattleAgent()        # Especialista en combate
        self.navigator_agent = NavigatorAgent()  # Especialista en navegación
        self.coordinator = CoordinatorAgent()    # Decide quién actúa
```

**4. Transfer Learning Entre Juegos:**
```python
# Transferir conocimiento a otros juegos de Pokémon
class TransferAgent:
    def __init__(self):
        self.base_policy = load_pokemon_red_policy()
        self.adaptation_layer = AdaptationNetwork()
    
    def adapt_to_new_game(self, new_game_env):
        # Reutilizar conocimiento previo para nuevo entorno
        pass
```

#### 🛠️ Implementaciones Técnicas Sugeridas:

**1. Análisis de Decisiones con Explicabilidad:**
```python
def explain_decision(state, action, model):
    """Explica por qué el agente tomó cierta decisión"""
    attention_weights = model.get_attention_weights(state)
    feature_importance = model.get_feature_importance(state)
    
    explanation = {
        "action_taken": action,
        "confidence": model.get_action_probability(state, action),
        "key_features": feature_importance.top_k(5),
        "attention_focus": attention_weights.max_regions(),
        "alternative_actions": model.get_top_k_actions(state, k=3)
    }
    return explanation
```

**2. Dynamic Reward Shaping:**
```python
class AdaptiveRewardSystem:
    def __init__(self):
        self.reward_weights = {"exploration": 0.25, "progress": 0.75}
        self.performance_history = []
    
    def update_reward_weights(self, recent_performance):
        """Ajusta pesos de recompensa según rendimiento"""
        if recent_performance["stuck_episodes"] > 5:
            self.reward_weights["exploration"] += 0.1
        if recent_performance["progress_rate"] < 0.1:
            self.reward_weights["progress"] += 0.1
```

**3. Curriculum Learning Explícito:**
```python
class PokemonCurriculum:
    def __init__(self):
        self.stages = [
            {"name": "basic_movement", "max_steps": 1000, "focus": "navigation"},
            {"name": "pokemon_capture", "max_steps": 5000, "focus": "interaction"},
            {"name": "battle_training", "max_steps": 10000, "focus": "combat"},
            {"name": "gym_challenges", "max_steps": 50000, "focus": "strategy"},
        ]
        self.current_stage = 0
    
    def should_advance_stage(self, agent_performance):
        current = self.stages[self.current_stage]
        return agent_performance.meets_criteria(current["focus"])
```

#### 📊 Métricas de Evaluación Innovadoras:

**1. Eficiencia de Exploración:**
```python
def calculate_exploration_efficiency(visited_coords, total_steps):
    """Mide qué tan eficientemente explora el agente"""
    unique_locations = len(set(visited_coords))
    return unique_locations / total_steps
```

**2. Consistencia de Política:**
```python
def measure_policy_consistency(actions_history, states_history):
    """Evalúa si el agente toma decisiones consistentes en estados similares"""
    similar_state_pairs = find_similar_states(states_history)
    consistency_score = 0
    for state1, state2 in similar_state_pairs:
        action1 = actions_history[state1]
        action2 = actions_history[state2]
        consistency_score += similarity(action1, action2)
    return consistency_score / len(similar_state_pairs)
```

**3. Adaptabilidad a Cambios:**
```python
def test_adaptability(agent, modified_environment):
    """Evalúa capacidad de adaptación a cambios en el entorno"""
    baseline_performance = agent.evaluate(original_environment)
    modified_performance = agent.evaluate(modified_environment)
    return modified_performance / baseline_performance
```

#### 🎯 Proyectos Específicos Sugeridos:

**1. "Agente Híbrido con Planificación Jerárquica"**
- Combinar PPO con A* para navegación
- Implementar planificación de objetivos
- Comparar eficiencia vs agente RL puro

**2. "Sistema de Explicabilidad para Decisiones RL"**
- Visualizar por qué el agente toma decisiones
- Implementar attention mechanisms
- Crear dashboard de explicaciones en tiempo real

**3. "Curriculum Learning Adaptativo"**
- Sistema que ajusta dificultad automáticamente
- Métricas de progreso personalizadas
- Comparación con entrenamiento estático

**4. "Multi-Agent Pokemon Ecosystem"**
- Múltiples agentes con roles especializados
- Comunicación entre agentes
- Estrategias emergentes de cooperación

#### 📝 Metodología de Investigación Sugerida:

**Fase 1: Análisis del Estado Actual**
1. Reproducir resultados de V2
2. Analizar limitaciones y puntos de mejora
3. Identificar casos donde falla el agente actual

**Fase 2: Diseño de Innovación**
1. Seleccionar área de mejora específica
2. Diseñar solución técnica detallada
3. Definir métricas de evaluación

**Fase 3: Implementación**
1. Desarrollar prototipo de la innovación
2. Integrar con sistema existente
3. Realizar pruebas comparativas

**Fase 4: Evaluación**
1. Comparar rendimiento vs baseline
2. Analizar trade-offs (tiempo vs precisión)
3. Documentar casos de uso donde la innovación es superior

#### 🏆 Criterios de Éxito del Proyecto:

1. **Mejora Medible**: ≥10% mejora en alguna métrica clave
2. **Innovación Técnica**: Implementación de técnica no utilizada previamente
3. **Aplicabilidad**: Solución generalizable a otros problemas RL
4. **Explicabilidad**: Capacidad de explicar por qué la solución funciona
5. **Reproducibilidad**: Código documentado y ejecutable por otrosdo## 🚨 Solución de Problemas

1. **Error de ROM**: Verificar que `PokemonRed.gb` esté en el directorio correcto con el hash correcto
2. **Problemas de dependencias**: Usar entorno virtual y versión específica de Python
3. **Errores de SDL**: Instalar bibliotecas SDL por separado si es necesario
4. **Rendimiento lento**: Considerar usar la versión V2 y ajustar `action_freq`
5. **Error PyBoy V2 - APIs obsoletas**: 
   - **Problema 1 - Memoria**: `AttributeError: 'PyBoy' object has no attribute 'memory'`
     ```python
     # En v2/red_gym_env_v2.py, línea ~461, cambiar:
     return self.pyboy.memory[addr]
     # A:
     return self.pyboy.get_memory_value(addr)
     ```
   - **Problema 2 - Pantalla**: `AttributeError: 'PyBoy' object has no attribute 'screen'`
     ```python
     # En v2/red_gym_env_v2.py, descomentar línea ~117:
     self.screen = self.pyboy.botsupport_manager().screen()
     # Y en línea ~171, cambiar:
     game_pixels_render = self.pyboy.screen.ndarray[:,:,0:1]
     # A:
     game_pixels_render = self.screen.screen_ndarray()[:,:,0:1]
     ```
   - **Problema 3 - Tick**: `TypeError: tick() takes exactly 0 positional arguments`
     ```python
     # En v2/red_gym_env_v2.py, método run_action_on_emulator, cambiar:
     self.pyboy.tick(press_step, render_screen)
     # A loops individuales:
     for i in range(press_step):
         self.pyboy.tick()
     ```
   - **Nota**: Estos errores ya están corregidos en este repositorio
6. **Error "Could not deserialize object tensorboard_log"**: 
   - Es una advertencia, no afecta la ejecución
   - Relacionado con compatibilidad de rutas entre Windows y sistemas Unixuego, capturar Pokémon, luchar en batallas y completar objetivos usando técnicas de aprendizaje profundo.

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
