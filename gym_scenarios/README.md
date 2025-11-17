<<<<<<< Updated upstream
# 🏟️ Sistema de Evaluación de Gimnasios Pokémon para Agentes PPO

Sistema completo para evaluar y comparar agentes PPO (base vs reentrenado) en los 8 gimnasios de Pokémon Red, midiendo desempeño en puzzles y combates.

## 📋 Tabla de Contenidos

- [Características](#características)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Gimnasios Configurados](#gimnasios-configurados)
- [Instalación](#instalación)
- [Uso Rápido](#uso-rápido)
- [Uso Detallado](#uso-detallado)
- [Métricas Capturadas](#métricas-capturadas)
- [Direcciones de Memoria](#direcciones-de-memoria)
- [Personalización](#personalización)

---

## ✨ Características

- **8 Escenarios de Gimnasios**: Uno por cada gimnasio de Pokémon Red
- **Equipos Realistas**: Configurados con Pokémon y niveles apropiados para cada gimnasio
- **Equipos Estratégicos**: Inicial débil al tipo del gimnasio para mayor desafío
- **Métricas Completas**: Tiempo, pasos, recompensas, combates, puzzles, navegación
- **Comparación Automática**: PPO Base vs PPO Reentrenado
- **Reportes Detallados**: JSON, CSV y Markdown
- **Totalmente Configurable**: Modificar equipos, items, posiciones fácilmente

---

## 📁 Estructura del Proyecto

```
gym_scenarios/
├── README.md                          # Este archivo
├── gym_memory_addresses.py            # Direcciones RAM de Pokémon Red
├── gym_metrics.py                     # Sistema de métricas
├── generate_gym_states.py             # Generador de archivos .state
├── run_gym_comparison.py              # Script principal de comparación
│
├── gym1_pewter_brock/                 # Gimnasio 1: Brock (Roca)
│   ├── team_config.json               # Configuración del equipo
│   └── gym_scenario.state             # Estado del juego (generado)
│
├── gym2_cerulean_misty/               # Gimnasio 2: Misty (Agua)
│   ├── team_config.json
│   └── gym_scenario.state
│
├── gym3_vermilion_lt_surge/           # Gimnasio 3: Lt. Surge (Eléctrico)
│   ├── team_config.json
│   └── gym_scenario.state
│
├── gym4_celadon_erika/                # Gimnasio 4: Erika (Planta)
│   ├── team_config.json
│   └── gym_scenario.state
│
├── gym5_fuchsia_koga/                 # Gimnasio 5: Koga (Veneno)
│   ├── team_config.json
│   └── gym_scenario.state
│
├── gym6_saffron_sabrina/              # Gimnasio 6: Sabrina (Psíquico)
│   ├── team_config.json
│   └── gym_scenario.state
│
├── gym7_cinnabar_blaine/              # Gimnasio 7: Blaine (Fuego)
│   ├── team_config.json
│   └── gym_scenario.state
│
└── gym8_viridian_giovanni/            # Gimnasio 8: Giovanni (Tierra)
    ├── team_config.json
    └── gym_scenario.state
```

---

## 🏟️ Gimnasios Configurados

| # | Gimnasio | Líder | Tipo | Puzzle | Inicial | Nivel | Dificultad |
|---|----------|-------|------|--------|---------|-------|------------|
| 1 | Pewter City | Brock | Roca | Ninguno | Charmander | 8-12 | Fácil |
| 2 | Cerulean City | Misty | Agua | Ninguno | Charmeleon | 14-18 | Media |
| 3 | Vermilion City | Lt. Surge | Eléctrico | Botes de basura | Wartortle | 21-25 | Media |
| 4 | Celadon City | Erika | Planta | Ninguno | Ivysaur | 27-30 | Media |
| 5 | Fuchsia City | Koga | Veneno | Paredes invisibles | Blastoise | 35-38 | Difícil |
| 6 | Saffron City | Sabrina | Psíquico | Teletransportadores | Charizard | 38-42 | Muy Difícil |
| 7 | Cinnabar Island | Blaine | Fuego | Quiz doors | Venusaur | 42-45 | Difícil |
| 8 | Viridian City | Giovanni | Tierra | Trainers y barreras | Blastoise | 45-50 | Muy Difícil |

### Estrategia de Equipos

Cada equipo está diseñado con:

1. **Starter débil al tipo del gimnasio** (e.g., Charmander vs Brock)
2. **Pokémon de soporte** apropiados para el nivel
3. **Niveles realistas** para cada punto del juego
4. **Items adecuados** (pociones, curas, pokéballs)
5. **Dinero razonable** para ese momento del juego

---

## 🔧 Instalación

### Requisitos Previos

```bash
# Python 3.8+
# PyBoy
# Stable-Baselines3
# NumPy, JSON, CSV
```

### Instalar Dependencias

```bash
cd gym_scenarios
pip install -r ../v2/requirements.txt
```

### Verificar ROM

Asegúrate de tener `PokemonRed.gb` en el directorio raíz del proyecto.

---

## 🚀 Uso Rápido

### 1. Generar Estados de Gimnasios

```bash
cd gym_scenarios
python generate_gym_states.py
```

Esto crea los 8 archivos `.state` con equipos configurados.

### 2. Evaluar un Gimnasio

```bash
# Evaluar Gimnasio 1 (Brock)
python run_gym_comparison.py --gym 1

# Evaluar todos los gimnasios
python run_gym_comparison.py --all
```

### 3. Ver Resultados

Los resultados se guardan en cada carpeta de gimnasio:

```
gym1_pewter_brock/results/
├── PPO_Base_gym1_<timestamp>_full.json
├── PPO_Base_gym1_<timestamp>_summary.csv
├── PPO_Base_gym1_<timestamp>_report.md
├── PPO_Retrained_gym1_<timestamp>_full.json
├── PPO_Retrained_gym1_<timestamp>_summary.csv
├── PPO_Retrained_gym1_<timestamp>_report.md
└── comparison_<timestamp>.json
```

---

## 📚 Uso Detallado

### Generar Estados Personalizados

```bash
# Generar todos los estados
python generate_gym_states.py

# El script usa team_config.json de cada carpeta
```

### Comparar Modelos Específicos

```bash
python run_gym_comparison.py \
    --gym 3 \
    --model-base path/to/base_model.zip \
    --model-retrained path/to/retrained_model.zip \
    --max-steps 15000
```

### Opciones del Script de Comparación

```bash
python run_gym_comparison.py --help

Opciones:
  --gym N              Número del gimnasio (1-8)
  --all                Evaluar todos los 8 gimnasios
  --model-base PATH    Path al modelo PPO base
  --model-retrained PATH  Path al modelo PPO reentrenado
  --headless           Ejecutar sin interfaz gráfica (default: True)
  --max-steps N        Máximo de pasos por episodio (default: 10000)
```

### Usar el Sistema de Métricas Manualmente

```python
from gym_metrics import GymMetricsTracker

# Crear tracker
tracker = GymMetricsTracker(
    gym_number=1,
    agent_name="PPO_Custom",
    gym_name="Pewter Gym"
)

# Iniciar
tracker.start()

# Durante ejecución
for step in range(1000):
    # ... ejecutar paso ...
    tracker.record_step(action, reward, game_state)
    
    # Eventos especiales
    if battle_started:
        tracker.record_battle_start()
    if puzzle_solved:
        tracker.record_puzzle_solved()

# Finalizar
tracker.end(success=True)
tracker.save_metrics()
```

---

## 📊 Métricas Capturadas

### Métricas de Tiempo
- Duración total (segundos)
- Pasos totales
- Pasos por segundo

### Métricas de Recompensa
- Recompensa total
- Recompensa promedio por paso
- Recompensa máxima/mínima

### Métricas de Combate
- Batalla ganada/perdida
- Duración del combate (pasos)
- Pokémon derrotados (jugador/oponente)
- Daño infligido/recibido

### Métricas de Puzzle
- Puzzle resuelto (sí/no)
- Intentos de resolución
- Pasos para resolver

### Métricas de Navegación
- Baldosas únicas exploradas
- Veces atascado
- Conteo de retrocesos

### Métricas de Items
- Pociones usadas
- Curas de estado usadas
- Total de items usados

### Métricas de Equipo
- HP promedio inicial
- HP promedio final
- Niveles del equipo

### Distribución de Acciones
- Frecuencia de cada acción (↑↓←→AB)
- Porcentaje de uso

---

## 🗺️ Direcciones de Memoria

El archivo `gym_memory_addresses.py` contiene todas las direcciones RAM necesarias:

### Equipo Pokémon
```python
PARTY_SIZE_ADDRESS = 0xD163
PARTY_ADDRESSES = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
LEVELS_ADDRESSES = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
HP_ADDRESSES = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
MAX_HP_ADDRESSES = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
```

### Items y Mochila
```python
BAG_ITEM_COUNT = 0xD31D
BAG_ITEMS_START = 0xD31E

ITEM_IDS = {
    'potion': 0x14,
    'super_potion': 0x15,
    'hyper_potion': 0x16,
    'antidote': 0x18,
    'poke_ball': 0x04,
    # ... más items
}
```

### Medallas
```python
BADGE_COUNT_ADDRESS = 0xD356

BADGE_BITS = {
    'boulder': 0,    # Brock
    'cascade': 1,    # Misty
    'thunder': 2,    # Lt. Surge
    'rainbow': 3,    # Erika
    'soul': 4,       # Koga
    'marsh': 5,      # Sabrina
    'volcano': 6,    # Blaine
    'earth': 7,      # Giovanni
}
```

### Posición
```python
X_POS_ADDRESS = 0xD362
Y_POS_ADDRESS = 0xD361
MAP_N_ADDRESS = 0xD35E

GYM_MAP_IDS = {
    'pewter': 54,
    'cerulean': 65,
    'vermilion': 92,
    'celadon': 123,
    'fuchsia': 146,
    'saffron': 178,
    'cinnabar': 166,
    'viridian': 45,
}
```

### IDs de Pokémon
```python
POKEMON_IDS = {
    'charmander': 0xB0,
    'charmeleon': 0xB2,
    'charizard': 0xB4,
    'squirtle': 0xB1,
    'wartortle': 0xB3,
    'blastoise': 0x1C,
    'bulbasaur': 0x99,
    'ivysaur': 0x09,
    'venusaur': 0x9A,
    # ... 150+ Pokémon
}
```

Ver `gym_memory_addresses.py` para la lista completa.

---

## 🎨 Personalización

### Modificar Equipo de un Gimnasio

Edita `team_config.json` en la carpeta del gimnasio:

```json
{
  "gym_name": "Pewter City Gym - Brock",
  "gym_number": 1,
  "player_team": [
    {
      "slot": 1,
      "species": "charmander",
      "species_id": 176,
      "level": 12,
      "current_hp": 33,
      "max_hp": 33,
      "moves": [
        {"name": "Scratch", "id": 10, "pp": 35},
        {"name": "Ember", "id": 52, "pp": 25}
      ]
    }
  ],
  "bag_items": [
    {"item": "potion", "item_id": 20, "quantity": 5}
  ],
  "money": 3000,
  "badges_before": [],
  "badge_bits": 0
}
```

Luego regenera el estado:

```bash
python generate_gym_states.py
```

### Crear Escenario Personalizado

1. Copia una carpeta de gimnasio existente
2. Modifica `team_config.json`
3. Actualiza `gym_memory_addresses.py` si necesitas nuevas direcciones
4. Ejecuta `generate_gym_states.py`

### Modificar Métricas

Edita `gym_metrics.py` para agregar nuevas métricas:

```python
class GymMetricsTracker:
    def __init__(self, ...):
        # Agregar nueva métrica
        self.my_custom_metric = 0
    
    def record_step(self, ...):
        # Actualizar métrica
        self.my_custom_metric += some_value
```

---

## 📖 Ejemplos de Uso

### Ejemplo 1: Evaluar Gimnasio Individual

```bash
# Evaluar solo el gimnasio de Brock
python run_gym_comparison.py --gym 1 --max-steps 8000
```

### Ejemplo 2: Evaluar Todos con Modelos Personalizados

```bash
python run_gym_comparison.py \
    --all \
    --model-base ../v2/my_base_model.zip \
    --model-retrained ../v2/my_retrained_model.zip
```

### Ejemplo 3: Análisis Programático

```python
import json
from pathlib import Path

# Cargar resultados de comparación
comparison_file = Path("gym1_pewter_brock/results/comparison_1234567890.json")
with open(comparison_file) as f:
    data = json.load(f)

# Analizar diferencias
for metric, diff in data['differences'].items():
    if diff.get('improved', False):
        print(f"{metric}: Mejoró {diff['percent_change']:.1f}%")
```

---

## 🔍 Solución de Problemas

### Error: ROM no encontrado

```bash
✗ ROM no encontrado: PokemonRed.gb
```

**Solución**: Coloca `PokemonRed.gb` en el directorio raíz del proyecto.

### Error: Modelo no encontrado

```bash
❌ Error: Modelo no encontrado: path/to/model.zip
```

**Solución**: Verifica que el path al modelo sea correcto o usa los defaults.

### Error: Estado no generado

```bash
⚠️  Advertencia: Estado no encontrado para gimnasio 1
```

**Solución**: Ejecuta primero `python generate_gym_states.py`.

---

## 📝 Notas Importantes

1. **Equipos Débiles**: Los equipos están diseñados intencionalmente con el starter débil al tipo del gimnasio para crear un desafío realista.

2. **Niveles Realistas**: Los niveles corresponden a lo que un jugador promedio tendría en cada punto del juego.

3. **Puzzles**: Algunos gimnasios tienen puzzles (botes de basura, teletransportadores) que el agente debe resolver.

4. **Detección de Eventos**: La detección de batalla ganada/puzzle resuelto actualmente requiere mejoras (marcado como TODO en el código).

5. **Archivos .state**: Los archivos `.state` son binarios de PyBoy y contienen el estado completo de la RAM del juego.

---

## 🤝 Contribuir

Para agregar nuevos escenarios o mejorar las métricas:

1. Crea nuevos archivos `team_config.json`
2. Agrega direcciones de memoria necesarias en `gym_memory_addresses.py`
3. Extiende `gym_metrics.py` con nuevas métricas
4. Actualiza `generate_gym_states.py` si necesitas lógica especial

---

## 📚 Referencias

- **Pokémon Red RAM Map**: https://datacrystal.tcrf.net/wiki/Pokémon_Red/Blue:RAM_map
- **PyBoy Documentation**: https://docs.pyboy.dk/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/

---

## 📄 Licencia

Este proyecto es parte del repositorio TEL351-PokemonRed. Ver LICENSE en el directorio raíz.

---

## ✅ Checklist de Uso

- [ ] ROM `PokemonRed.gb` en directorio raíz
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Estados generados (`python generate_gym_states.py`)
- [ ] Modelos PPO disponibles
- [ ] Ejecutar comparación (`python run_gym_comparison.py --all`)
- [ ] Revisar resultados en carpetas `results/`

---

**¡Listo para comparar tus agentes PPO en los 8 gimnasios de Pokémon Red! 🎮🔥**
=======
# Gym Scenario Harness

This module adds structure to define and benchmark eight stress-test scenarios (one per Kanto gym) using the existing PPO environment. It does **not** ship proprietary `.state` snapshots; instead it provides the scaffolding, file names, and tooling you need to capture them locally and run side-by-side evaluations of two PPO checkpoints.

## Repository Additions

```
gym_scenarios/
├── evaluate_agents.py     # Batch runner that compares baseline vs improved PPO over every phase
├── scenarios.json         # Metadata for each gym (teams, items, badge bit, state filenames, goals)
├── state_files/           # Drop your captured .state files here (placeholders only)
│   └── .gitkeep           # Keeps the directory under version control
└── README.md              # This guide
```

## Scenario Metadata (`scenarios.json`)
* `state_directory` – relative path where state files must live.
* `default_max_steps` – fallback horizon for any phase that does not override `max_steps`.
* `scenarios[]` – one entry per gym:
  * `id`, `leader`, `city`, `badge_bit`, `map_id` – identifiers and RAM context.
  * `starter`, `recommended_party`, `inventory` – documentation so you can rebuild realistic teams/items.
  * `phases[]` – two phases per gym (`puzzle`, `battle`). Each phase lists the `.state` file name, goal configuration, and per-phase `max_steps`.
  * `goal.type` can be `badge`, `coordinate`, `manual`, or `none`. Battle phases already watch the badge bit. Puzzle phases default to `manual` until you add coordinate windows once the states exist.

## Required `.state` Files
Capture the following 16 files (two per gym) and drop them under `gym_scenarios/state_files/`:

| Gym | Puzzle State | Battle State | Notes |
| --- | --- | --- | --- |
| Pewter / Boulder | `pewter_puzzle.state` | `pewter_battle.state` | Entrance tile and pre-dialog vs Brock |
| Cerulean / Cascade | `cerulean_puzzle.state` | `cerulean_battle.state` | After clearing bridge / before Misty |
| Vermilion / Thunder | `vermilion_puzzle.state` | `vermilion_battle.state` | Trash-can puzzle solved vs Lt. Surge |
| Celadon / Rainbow | `celadon_puzzle.state` | `celadon_battle.state` | Gym ring cleared |
| Fuchsia / Soul | `fuchsia_puzzle.state` | `fuchsia_battle.state` | Invisible-wall maze |
| Saffron / Marsh | `saffron_puzzle.state` | `saffron_battle.state` | Teleporter maze |
| Cinnabar / Volcano | `cinnabar_puzzle.state` | `cinnabar_battle.state` | Quiz terminal path |
| Viridian / Earth | `viridian_puzzle.state` | `viridian_battle.state` | Final conveyor + Giovanni dialog |

### How to Record a State
1. Launch an interactive PPO session (e.g. `python v2/run_pretrained_interactive.py`) but set `init_state` in the config to a recent checkpoint such as `has_pokedex.state` or the previous gym battle.
2. Play manually until you reach the exact tile requested (entrance of puzzle or dialog box before the leader). Make sure the badge corresponding to that gym is still **unset** so the badge-goal check is meaningful.
3. Pause the game and press `Shift+F1` inside PyBoy to save the emulator snapshot, or call `pyboy.save_state` from the debugger. Save the file using the names listed above and move it into `gym_scenarios/state_files/`.
4. (Optional) Update the `goal` section for the puzzle phase with coordinate bounds once you know the `map_id`, `x`, and `y` values from RAM (see table below). That enables automatic success detection.

## RAM Reference Cheat-Sheet
The most common addresses you need when crafting or validating states are already exposed by `RedGymEnv.read_m`. The table below consolidates the ones used for teams, inventory, and goals; all values are hexadecimal offsets inside WRAM bank 0.

| Purpose | Address | Description |
| --- | --- | --- |
| Party size | `0xD163` | Number of active Pokémon in party |
| Party species list | `0xD164`–`0xD169` | Species IDs for each slot (0xFF terminator) |
| Party levels | `0xD18C`, `0xD1B8`, `0xD1E4`, `0xD210`, `0xD23C`, `0xD268` | Level byte per party slot (already used by the env) |
| Party current HP | `0xD16C`, `0xD198`, `0xD1C4`, `0xD1F0`, `0xD21C`, `0xD248` | Two bytes per slot (hi, lo) |
| Party max HP | `0xD18D`, `0xD1B9`, `0xD1E5`, `0xD211`, `0xD23D`, `0xD269` | Two bytes per slot |
| Bag items | `0xD31C` onward | Each entry is `[item_id, quantity]` until `0xFF` terminator |
| Money | `0xD347`–`0xD349` | BCD encoded ¥ amount |
| Badge flags | `0xD356` | Each bit corresponds to a badge (0=Boulder,…,7=Earth) |
| Player X/Y | `0xD362` / `0xD361` | Local tile coordinates |
| Map ID | `0xD35E` | Matches `map_id` field inside `scenarios.json` |
| Encounter/battle flag | `0xD057` | Non-zero when in battle (useful when validating puzzle end conditions) |

You can patch data inside a state before saving it permanently by assigning to `env.pyboy.memory[address]`. For better reproducibility, prefer editing `scenarios.json` and keeping a short note in the `inventory.notes` or `starter.disadvantage` fields so that collaborators know the intended composition when they reproduce the `.state` file.

## Running Evaluations
1. Train or collect two PPO checkpoints (the “baseline” and “improved” models). Both must be `.zip` files compatible with `stable_baselines3.PPO.load`.
2. Drop your sixteen `.state` files into `gym_scenarios/state_files/`.
3. Run the batch script:

```bash
python gym_scenarios/evaluate_agents.py \
  --baseline v2/runs/poke_baseline.zip \
  --improved v2/runs/poke_improved.zip \
  --episodes 5 \
  --deterministic \
  --gb-path PokemonRed.gb
```

* Use `--windowed` if you want to watch the rollouts instead of running headless.
* Results land under `gym_scenarios/results/<timestamp>/<scenario>/<phase>/<agent>.json`. Each JSON records per-episode metrics (steps, reward, wall-clock duration, success flag, final coordinates) plus an averaged summary.
* A top-level `summary.json` aggregates everything so you can feed it into your reporting or visualization pipeline.

## What Still Requires Manual Work
* The repository cannot ship the actual `.state` files for legal reasons and because they depend on your ROM hash. Use the instructions above to capture them locally.
* Puzzle-phase goals are marked as `manual` by default. After you capture each state, update the `goal` object with the observed `(map_id, x_range, y_range)` so success detection no longer needs human review.
* If you change the recommended party or inventory for a scenario, regenerate the `.state` file (or patch the relevant RAM addresses) so that both PPO checkpoints see the same initial conditions.

With these pieces in place you can now stress-test any PPO iteration by running the evaluation script and comparing steps-to-success, reward accumulation, and badge acquisition across the eight canonical gyms.
>>>>>>> Stashed changes
