# Epsilon Greedy Search Algorithm

Esta carpeta contiene la implementación del algoritmo **Epsilon Greedy** para Pokemon Red.

## ¿Por qué Epsilon Greedy?

El algoritmo **Eps#### Uso del Triple Demo Mejorado:
```bash
# Ejecutar las tres demos simultáneamente con identificación múltiple
python epsilon_greedy/run_triple_epsilon_demo.py

# Verás 3 ventanas CLARAMENTE IDENTIFICADAS:
# Izquierda: "EPSILON 0.3 MODERADO" + [EPSILON-0.3-MODERADO]
# Centro: "EPSILON 0.9 CAOTICO" + [EPSILON-0.9-CAOTICO]
# Derecha: "EPSILON VARIABLE INTERACTIVO" + [EPSILON-VARIABLE-INTERACTIVO]
```

#### Identificación Sin Elementos Visuales:
Si los títulos o prefijos no son visibles, consulta:
- **[GUIA_IDENTIFICACION_VISUAL.md](GUIA_IDENTIFICACION_VISUAL.md)** - Métodos para distinguir agentes por comportamiento
- **[VIDEO_GUION_COMPARACION_AGENTES.md](VIDEO_GUION_COMPARACION_AGENTES.md)** - Análisis completo y justificación de elección

**Resumen Rápido de Identificación:**
- **Epsilon 0.9**: Movimientos completamente erráticos y caóticos (el MÁS FÁCIL de identificar)
- **PPO**: Movimientos robóticos y súper eficientes (línea directa a objetivos)
- **Epsilon 0.3**: Balance visible entre exploración y direccionalidad
- **Adaptativo**: Comportamiento "inteligente" que cambia según el contexto

Esta funcionalidad hace que la comparación visual sea mucho más intuitiva y educativa.* fue seleccionado para este proyecto por las siguientes razones:

### Ventajas del Epsilon Greedy
- **Simplicidad**: Es un algoritmo fácil de entender e implementar
- **Balance exploración-explotación**: Permite controlar el balance entre explorar nuevas acciones y explotar el conocimiento actual
- **Adaptabilidad**: El parámetro epsilon se puede ajustar según las necesidades del entorno
- **Eficiencia computacional**: No requiere estructuras de datos complejas ni cálculos intensivos
- **Robustez**: Funciona bien en entornos con recompensas escasas como Pokemon Red

### Comparación con otros algoritmos
- **A*** requiere conocimiento previo del entorno y función heurística, que no siempre está disponible
- **Tabú Search** necesita memoria adicional para rastrear estados visitados y puede ser más lento
- **Algoritmos evolutivos** requieren poblaciones y múltiples generaciones, siendo computacionalmente más costosos
- **Q-Learning/DQN** necesitan entrenamiento extenso y pueden requerir más recursos

En Pokemon Red, donde el objetivo es navegar eficientemente sin conocimiento previo completo del entorno, Epsilon Greedy ofrece una solución práctica y efectiva.

## Comparación con PPO Preentrenado

### ¿Por qué Epsilon Greedy Adaptativo sobre PPO?

Aunque el **PPO preentrenado** (disponible en `/v2`) es **más rápido** para completar Pokemon Red, el **Epsilon Greedy Adaptativo** ofrece ventajas fundamentales:

#### Ventajas del PPO:
- **Velocidad**: Completa objetivos en 2-4 minutos
- **Eficiencia**: Movimientos optimizados y directos
- **Rendimiento**: Maximiza recompensas acumulativas

#### Limitaciones del PPO:
- **Dependiente de entrenamiento**: Requiere miles de episodios previos
- **Caja negra**: Difícil explicar decisiones específicas
- **Especialización**: Optimizado solo para Pokemon Red específico
- **Costo computacional**: Entrenamiento requiere recursos significativos
- **Falta de adaptabilidad**: No se ajusta a variaciones del entorno

#### Ventajas del Epsilon Greedy Adaptativo:
- **Sin entrenamiento**: Funciona inmediatamente "out of the box"
- **Transparencia**: Cada decisión es explicable mediante heurísticas
- **Generalizable**: Se adapta a Pokemon Blue, hacks, modificaciones
- **Robusto**: Funciona en situaciones no vistas previamente
- **Comportamiento humano**: Decisiones similares a las que tomaría un jugador
- **Modificable**: Fácil ajustar heurísticas según necesidades

#### El Factor Crucial: Generalización vs Optimización
- **PPO**: Optimizado para Pokemon Red → Rápido pero especializado
- **Epsilon Adaptativo**: Principios generales → Robusto y adaptable

**Conclusión**: Elegimos Epsilon Greedy Adaptativo porque prioriza **robustez, transparencia y generalización** sobre velocidad pura, principios fundamentales en investigación y desarrollo de IA.

## Adaptabilidad del Epsilon: Funcionamiento Detallado

### Cómo opera la adaptabilidad en `run_epsilon_greedy_interactive.py`

El agente epsilon greedy implementa un sistema de **adaptabilidad multicapa** que ajusta el comportamiento según el contexto del juego:

#### 1. **Detección de Escenarios en Tiempo Real**
El agente detecta automáticamente el escenario actual analizando la observación del juego:

```python
# Escenarios detectados automáticamente:
- EXPLORATION: Exploración general del mapa
- BATTLE: Combate activo 
- NAVIGATION: Navegación hacia objetivos específicos
- PROGRESSION: Eventos clave (gimnasios, captura de Pokémon)
- STUCK: Detección de comportamiento repetitivo
```

#### 2. **Heurísticas Adaptativas por Escenario**
Cada escenario tiene pesos heurísticos específicos que priorizan diferentes comportamientos:

**Exploración** (EXPLORATION):
- Prioriza descubrir nuevas áreas (exploration=1.5)
- Moderado enfoque en objetivos (objective_distance=0.8)
- Balance en todas las métricas

**Combate** (BATTLE):
- Máxima prioridad en salud (health_consideration=2.0)
- Enfoque en progresión de nivel (level_progression=1.5)
- Mínima exploración (exploration=0.2)

**Navegación** (NAVIGATION):
- Máxima prioridad en distancia a objetivos (objective_distance=2.0)
- Alta familiaridad con el mapa (map_familiarity=1.2)
- Exploración reducida pero presente

**Progresión** (PROGRESSION):
- Máxima prioridad en eventos clave (event_completion=2.5)
- Alta priorización de nivel (level_progression=2.0)
- Balance entre exploración y explotación

**Atascado** (STUCK):
- Máxima exploración forzada (exploration=2.0)
- Penalización de familiaridad (map_familiarity=0.3)
- Fomenta salir de patrones repetitivos

#### 3. **Cálculo Dinámico de Epsilon**
Aunque epsilon tiene un valor base, el comportamiento efectivo cambia según el escenario:

```python
# Proceso de selección de acción:
if random() < epsilon:
    # EXPLORACIÓN: Acción aleatoria
    action = random_choice(valid_actions)
else:
    # EXPLOTACIÓN: Mejor acción según heurísticas del escenario actual
    action = best_heuristic_action(current_scenario)
```

#### 4. **Decay Automático y Progresivo**
El epsilon se reduce gradualmente durante la ejecución:

```python
# Después de cada acción:
epsilon = max(epsilon_min, epsilon * epsilon_decay)
# Típicamente: epsilon_decay = 0.995, epsilon_min = 0.05
```

#### 5. **Detección Inteligente de Progreso**
El sistema monitorea indicadores de progreso para ajustar el comportamiento:

- **Badges obtenidas**: Detecta completar gimnasios
- **Eventos del juego**: Captura de Pokémon, obtención de ítems
- **Suma de niveles**: Progreso en entrenamiento
- **Datos del mapa**: Nuevas áreas exploradas
- **Tamaño del equipo**: Pokémon en el equipo

#### 6. **Sistema Anti-Estancamiento**
Detecta cuando el agente está repitiendo comportamientos:

```python
# Detección de bucles:
- Mismas posiciones repetidas
- Mismas acciones en secuencia
- Falta de progreso en métricas clave

# Respuesta automática:
- Cambio a escenario STUCK
- Aumento temporal de exploración
- Penalización de acciones familiares
```

### Ventajas de este Sistema Adaptativo

1. **Respuesta contextual**: El agente se comporta diferente en combate vs exploración
2. **Prevención de bucles**: Detecta y rompe patrones repetitivos automáticamente  
3. **Optimización progresiva**: Epsilon decrece conforme el agente "aprende" el entorno
4. **Sin entrenamiento previo**: Funciona inmediatamente sin necesidad de entrenamiento
5. **Transparencia**: Cada decisión se basa en heurísticas comprensibles

### Diferencia con Epsilon Fijo

**Epsilon Fijo** (demos):
- Comportamiento constante durante toda la ejecución
- No se adapta al contexto del juego
- Útil para benchmarks y comparaciones

**Epsilon Adaptativo** (interactive):
- Comportamiento dinámico según el escenario
- Se adapta a la situación del juego
- Optimiza rendimiento según el contexto
- Previene estancamiento automáticamente

Esta adaptabilidad hace que `run_epsilon_greedy_interactive.py` sea más inteligente y eficiente que una implementación con epsilon fijo, permitiendo un comportamiento más humano y estratégico.

## Identificación Visual MEJORADA de Ventanas PyBoy

### Triple Demo con Identificación Visual Múltiple y Robusta

El script `run_triple_epsilon_demo.py` ejecuta simultáneamente tres agentes epsilon greedy con **múltiples métodos de identificación visual** para eliminar cualquier confusión:

#### 1. Títulos de Ventana Súper Descriptivos:
- **Esquina Superior Izquierda**: `"POKEMON RED ===>>> EPSILON 0.3 MODERADO <<<==== 30-EXPLORA 70-EXPLOTA"`
  - 30% exploración, 70% explotación  
  - Comportamiento balanceado y estratégico

- **Centro Superior**: `"POKEMON RED ===>>> EPSILON 0.9 CAOTICO <<<==== 90-EXPLORA 10-EXPLOTA"`
  - 90% exploración, 10% explotación
  - Comportamiento muy exploratorio y aleatorio

- **Esquina Superior Derecha**: `"POKEMON RED ===>>> EPSILON VARIABLE INTERACTIVO <<<==== ADAPTATIVO"`
  - Epsilon adaptativo según escenario
  - Comportamiento que puedes modificar en tiempo real

#### 2. Posicionamiento Automático de Ventanas:
- **Demo 0.3**: Posición (100, 100) - Esquina superior izquierda
- **Demo 0.9**: Posición (500, 100) - Centro superior  
- **Demo Variable**: Posición (900, 100) - Esquina superior derecha

#### 3. Prefijos Únicos en Salida de Consola:
- **Demo 0.3**: `[EPSILON-0.3-MODERADO]` en todas las líneas de output
- **Demo 0.9**: `[EPSILON-0.9-CAOTICO]` en todas las líneas de output
- **Demo Variable**: `[EPSILON-VARIABLE-INTERACTIVO]` en todas las líneas de output

#### 4. Metadatos del Stream Diferenciados:
- **Epsilon 0.3**: ID `"E03"`, identifier `"MODERADO"`, behavior `"30% Exploración - 70% Explotación"`
- **Epsilon 0.9**: ID `"E09"`, identifier `"CAOTICO"`, behavior `"90% Exploración - 10% Explotación"`  
- **Epsilon Variable**: ID `"EVAR"`, identifier `"INTERACTIVO"`, behavior `"Adaptativo según escenario detectado"`

#### Ventajas de la Identificación Visual Mejorada:
1. **IMPOSIBLE DE CONFUNDIR**: Múltiples métodos redundantes de identificación
2. **Títulos súper largos**: Inmediatamente visibles en la barra de título
3. **Posiciones diferentes**: Cada ventana aparece en ubicación específica
4. **Prefijos en consola**: Cada output tiene su identificador único
5. **Comparación fácil**: Observa diferencias de comportamiento simultáneamente
6. **Sin dependencia de colores**: Funciona incluso si los colores no se muestran

#### Uso del Triple Demo Mejorado:
```bash
# Ejecutar las tres demos simultáneamente con identificación múltiple
python epsilon_greedy/run_triple_epsilon_demo.py

# Verás 3 ventanas CLARAMENTE IDENTIFICADAS:
# � Izquierda: "EPSILON 0.3 MODERADO" + [EPSILON-0.3-MODERADO]
# � Centro: "EPSILON 0.9 CAOTICO" + [EPSILON-0.9-CAOTICO]
# � Derecha: "EPSILON VARIABLE INTERACTIVO" + [EPSILON-VARIABLE-INTERACTIVO]
```

Esta funcionalidad hace que la comparación visual sea mucho más intuitiva y educativa.

## Archivos

- `epsilon_greedy_agent.py` - Implementación original del agente epsilon greedy que funciona muy bien
- `epsilon_variable_agent.py` - Versión avanzada con epsilon configurable para estudiar impacto en rendimiento
- `run_epsilon_greedy_interactive.py` - Script interactivo para ejecutar el agente original
- `demo_pyboy_epsilon_03.py` - Demo automático con epsilon fijo 0.3 (exploración moderada)
- `demo_pyboy_epsilon_09.py` - Demo automático con epsilon fijo 0.9 (exploración muy alta)
- `test_epsilon_variants.py` - Script para probar diferentes valores de epsilon

## Diferencias entre los Scripts Principales

### `run_epsilon_greedy_interactive.py`
- **Propósito**: Script principal interactivo para uso general
- **Epsilon**: Variable según el escenario detectado (adaptativo)
- **Control**: Permite interacción manual y control fino
- **Objetivo**: Ejecutar hasta obtener el primer Pokémon o elegir inicial
- **Uso**: Desarrollo y testing principal del algoritmo

### `demo_pyboy_epsilon_03.py`
- **Propósito**: Demostración automática con comportamiento balanceado
- **Epsilon**: Fijo en 0.3 (30% exploración, 70% explotación)
- **Control**: Completamente automático, sin intervención del usuario
- **Objetivo**: Mostrar comportamiento moderado y eficiente
- **Uso**: Demonstraciones y comparaciones de rendimiento

### `demo_pyboy_epsilon_09.py`
- **Propósito**: Demostración automática con comportamiento exploratorio extremo
- **Epsilon**: Fijo en 0.9 (90% exploración, 10% explotación)
- **Control**: Completamente automático, sin intervención del usuario
- **Objetivo**: Mostrar comportamiento altamente exploratorio y caótico
- **Uso**: Análisis de comportamiento exploratorio extremo

### Métricas Compartidas
Todos los scripts generan métricas idénticas en la carpeta `/results`:
- **Markdown**: Informes detallados con estadísticas completas
- **JSON**: Datos crudos para análisis programático
- **CSV**: Resumen de métricas para importar en hojas de cálculo

## Variaciones de Epsilon

El archivo `epsilon_variable_agent.py` incluye configuraciones predefinidas:

- **very_high_exploration** (ε=0.9): 90% exploración - casi aleatorio
- **high_exploration** (ε=0.7): 70% exploración - mucha exploración  
- **balanced** (ε=0.5): 50% exploración - enfoque balanceado
- **moderate_exploitation** (ε=0.3): 30% exploración - más explotación
- **low_exploration** (ε=0.1): 10% exploración - principalmente explotación
- **very_low_exploration** (ε=0.05): 5% exploración - casi pura explotación
- **pure_exploitation** (ε=0.01): 1% exploración - casi greedy

## Uso

### Ejecutar agente principal (adaptativo):
```bash
python epsilon_greedy/run_epsilon_greedy_interactive.py
```

### Ejecutar demo con epsilon moderado (0.3):
```bash
python epsilon_greedy/demo_pyboy_epsilon_03.py
```

### Ejecutar demo con epsilon alto (0.9):
```bash
python epsilon_greedy/demo_pyboy_epsilon_09.py
```

### Ejecutar triple demo simultáneo con identificación visual:
```bash
python epsilon_greedy/run_triple_epsilon_demo.py
```

### Probar diferentes valores de epsilon:
```bash
python epsilon_greedy/test_epsilon_variants.py
```

### Crear agente con epsilon específico:
```python
from epsilon_greedy.epsilon_variable_agent import VariableEpsilonGreedyAgent
agent = VariableEpsilonGreedyAgent(env, epsilon=0.3)  # 30% exploración
```

### Usar configuraciones predefinidas:
```python
from epsilon_greedy.epsilon_variable_agent import create_agent_with_preset
agent = create_agent_with_preset(env, 'balanced')  # ε=0.5
```

## Rendimiento

El epsilon greedy original ha demostrado **excelente rendimiento** en Pokemon Red. Los experimentos con diferentes valores de epsilon permiten estudiar:

- **Alto epsilon** → Más exploración, descubre nuevas áreas pero puede ser errático
- **Bajo epsilon** → Más explotación, más eficiente pero puede quedarse atascado
- **Epsilon balanceado** → Combina exploración y explotación efectivamente