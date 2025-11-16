# 🔧 QUÉ CAMBIAR Y DÓNDE CAMBIARLO

Esta guía te indica **exactamente** qué archivos modificar para personalizar cada aspecto del sistema.

---

## 📍 Cambios Comunes

### 1. Cambiar el Equipo de un Gimnasio

**RUTA**: `gym_scenarios/gym{N}_{nombre}/team_config.json`

Ejemplo para Gimnasio 1:
```
📁 gym_scenarios/gym1_pewter_brock/team_config.json
```

**QUÉ CAMBIAR**:

```json
{
  "player_team": [
    {
      "slot": 1,
      "species": "charmander",      // ← Cambiar Pokémon
      "species_id": 176,            // ← Cambiar ID (ver tabla abajo)
      "level": 12,                  // ← Cambiar nivel
      "current_hp": 33,             // ← HP actual
      "max_hp": 33,                 // ← HP máximo
      "moves": [                    // ← Cambiar movimientos
        {"name": "Scratch", "id": 10, "pp": 35}
      ]
    }
  ]
}
```

**DESPUÉS DE CAMBIAR**:
```bash
python generate_gym_states.py
```

---

### 2. Cambiar Items Disponibles

**RUTA**: `gym_scenarios/gym{N}_{nombre}/team_config.json`

**QUÉ CAMBIAR**:

```json
{
  "bag_items": [
    {"item": "potion", "item_id": 20, "quantity": 5},        // ← Pociones
    {"item": "super_potion", "item_id": 21, "quantity": 3},  // ← Super pociones
    {"item": "hyper_potion", "item_id": 22, "quantity": 10}, // ← Añadir nuevo
    {"item": "revive", "item_id": 31, "quantity": 2}         // ← Revivir
  ]
}
```

**IDs de items comunes** (ver lista completa en `gym_memory_addresses.py`):
- Potion: `20`
- Super Potion: `21`
- Hyper Potion: `22`
- Max Potion: `23`
- Full Restore: `29`
- Revive: `31`
- Max Revive: `32`
- Antidote: `24`
- Paralyze Heal: `28`
- Full Heal: `30`

**DESPUÉS DE CAMBIAR**:
```bash
python generate_gym_states.py
```

---

### 3. Cambiar Dinero

**RUTA**: `gym_scenarios/gym{N}_{nombre}/team_config.json`

**QUÉ CAMBIAR**:

```json
{
  "money": 10000    // ← Cambiar cantidad
}
```

**DESPUÉS DE CAMBIAR**:
```bash
python generate_gym_states.py
```

---

### 4. Cambiar Medallas Previas

**RUTA**: `gym_scenarios/gym{N}_{nombre}/team_config.json`

**QUÉ CAMBIAR**:

```json
{
  "badges_before": ["boulder", "cascade", "thunder"],  // ← Medallas obtenidas
  "badge_bits": 7    // ← Binario: 0000 0111 = 3 medallas
}
```

**Cálculo de badge_bits**:
- Boulder (Brock) = bit 0 = 1
- Cascade (Misty) = bit 1 = 2
- Thunder (Surge) = bit 2 = 4
- Rainbow (Erika) = bit 3 = 8
- Soul (Koga) = bit 4 = 16
- Marsh (Sabrina) = bit 5 = 32
- Volcano (Blaine) = bit 6 = 64
- Earth (Giovanni) = bit 7 = 128

**Ejemplo**: Boulder + Cascade + Thunder = 1 + 2 + 4 = 7

**DESPUÉS DE CAMBIAR**:
```bash
python generate_gym_states.py
```

---

### 5. Cambiar Posición Inicial

**RUTA**: `gym_scenarios/gym{N}_{nombre}/team_config.json`

**QUÉ CAMBIAR**:

```json
{
  "map_id": 54,              // ← ID del mapa
  "start_position": {
    "x": 4,                  // ← Coordenada X
    "y": 13                  // ← Coordenada Y
  }
}
```

**IDs de mapas de gimnasios** (ver `gym_memory_addresses.py`):
- Pewter (Brock): `54`
- Cerulean (Misty): `65`
- Vermilion (Lt. Surge): `92`
- Celadon (Erika): `123`
- Fuchsia (Koga): `146`
- Saffron (Sabrina): `178`
- Cinnabar (Blaine): `166`
- Viridian (Giovanni): `45`

**DESPUÉS DE CAMBIAR**:
```bash
python generate_gym_states.py
```

---

### 6. Cambiar Paths de Modelos PPO

**RUTA**: `gym_scenarios/run_gym_comparison.py`

**LÍNEAS**: ~345-356

**QUÉ CAMBIAR**:

```python
parser.add_argument(
    '--model-base',
    type=str,
    default='../v2/MI_MODELO_BASE.zip',        # ← Cambiar aquí
    help='Path al modelo PPO base'
)

parser.add_argument(
    '--model-retrained',
    type=str,
    default='../v2/MI_MODELO_MEJORADO.zip',    # ← Cambiar aquí
    help='Path al modelo PPO reentrenado'
)
```

**NO NECESITAS REGENERAR NADA**, solo ejecuta de nuevo.

---

### 7. Cambiar Máximo de Pasos

**RUTA**: `gym_scenarios/run_gym_comparison.py`

**LÍNEAS**: ~358-363

**QUÉ CAMBIAR**:

```python
parser.add_argument(
    '--max-steps',
    type=int,
    default=15000,    // ← Cambiar de 10000 a lo que necesites
    help='Máximo de pasos por episodio'
)
```

O usar argumento de línea de comandos:
```bash
python run_gym_comparison.py --gym 1 --max-steps 20000
```

---

### 8. Agregar Nuevas Métricas

**RUTA**: `gym_scenarios/gym_metrics.py`

**PASO 1 - Inicializar variable** (líneas ~40-80):

```python
class GymMetricsTracker:
    def __init__(self, ...):
        # ... métricas existentes ...
        
        # TU NUEVA MÉTRICA
        self.mi_nueva_metrica = 0
        self.mi_contador = 0
```

**PASO 2 - Actualizar en cada paso** (líneas ~100-150):

```python
def record_step(self, action, reward, game_state):
    # ... código existente ...
    
    # ACTUALIZAR TU MÉTRICA
    if alguna_condicion:
        self.mi_nueva_metrica += valor
```

**PASO 3 - Incluir en resumen** (líneas ~250-290):

```python
def get_summary_stats(self):
    return {
        # ... métricas existentes ...
        
        # TU NUEVA MÉTRICA
        'mi_nueva_metrica': self.mi_nueva_metrica,
        'mi_contador': self.mi_contador,
    }
```

**PASO 4 - Incluir en reporte Markdown** (líneas ~450-500):

```python
def _save_markdown_report(self, path, data):
    # ... código existente ...
    
    report += f"""
## 📊 Mi Nueva Sección
- **Mi Métrica:** {summary['mi_nueva_metrica']}
"""
```

---

## 📊 Tablas de Referencia

### IDs de Pokémon (los más comunes)

Ver lista completa en `gym_scenarios/gym_memory_addresses.py` líneas 78-230.

```python
# Starters
'charmander': 0xB0      # 176
'charmeleon': 0xB2      # 178
'charizard': 0xB4       # 180
'squirtle': 0xB1        # 177
'wartortle': 0xB3       # 179
'blastoise': 0x1C       # 28
'bulbasaur': 0x99       # 153
'ivysaur': 0x09         # 9
'venusaur': 0x9A        # 154

# Comunes
'pidgey': 0x24          # 36
'pidgeotto': 0x96       # 150
'pidgeot': 0x97         # 151
'rattata': 0xA5         # 165
'raticate': 0xA6        # 166
'pikachu': 0x54         # 84
'raichu': 0x55          # 85
'geodude': 0xA9         # 169
'graveler': 0x27        # 39
'golem': 0x31           # 49
'onix': 0x22            # 34

# Evolucionados comunes
'alakazam': 0x95        # 149
'gyarados': 0x16        # 22
'arcanine': 0x14        # 20
'rhydon': 0x01          # 1
'lapras': 0x13          # 19
'dugtrio': 0x76         # 118
```

### IDs de Items

Ver lista completa en `gym_scenarios/gym_memory_addresses.py` líneas 36-65.

```python
# Curación
'potion': 0x14          # 20
'super_potion': 0x15    # 21
'hyper_potion': 0x16    # 22
'max_potion': 0x17      # 23
'full_restore': 0x1D    # 29
'revive': 0x1F          # 31
'max_revive': 0x20      # 32

# Curas de estado
'antidote': 0x18        # 24
'burn_heal': 0x19       # 25
'ice_heal': 0x1A        # 26
'awakening': 0x1B       # 27
'paralyze_heal': 0x1C   # 28
'full_heal': 0x1E       # 30

# Pokéballs
'poke_ball': 0x04       # 4
'great_ball': 0x05      # 5
'ultra_ball': 0x06      # 6

# Battle items
'x_attack': 0x31        # 49
'x_defend': 0x32        # 50
'x_speed': 0x33         # 51
'x_special': 0x34       # 52

# Otros
'escape_rope': 0x28     # 40
'repel': 0x29           # 41
```

---

## 🗺️ Direcciones de Memoria para Modificaciones Avanzadas

### Estructura de un Pokémon en RAM

Cada Pokémon ocupa 44 bytes (0x2C) en memoria:

```
Base: 0xD16B + (slot * 0x2C)

Offset 0x00: Species ID
Offset 0x01-0x02: Current HP (big endian, 2 bytes)
Offset 0x08: Level
Offset 0x21-0x22: Max HP (2 bytes)
Offset 0x23-0x24: Attack (2 bytes)
Offset 0x25-0x26: Defense (2 bytes)
Offset 0x27-0x28: Speed (2 bytes)
Offset 0x29-0x2A: Special (2 bytes)
```

**Ejemplo para Slot 1**:
```python
# Pokémon 1 empieza en 0xD16B
POKEMON_1_BASE = 0xD16B
POKEMON_1_SPECIES = 0xD164    # Species ID
POKEMON_1_HP = 0xD16C         # Current HP (2 bytes)
POKEMON_1_LEVEL = 0xD18C      # Level
POKEMON_1_MAX_HP = 0xD18D     # Max HP (2 bytes)
```

### Movimientos

Cada Pokémon tiene 4 movimientos:

```python
MOVES_PP_SLOT_1 = [0xD188, 0xD189, 0xD18A, 0xD18B]
MOVES_PP_SLOT_2 = [0xD1B4, 0xD1B5, 0xD1B6, 0xD1B7]
# ... etc
```

### Items en Mochila

Formato en memoria:
```
0xD31D: Número de items (N)
0xD31E: Item 1 ID
0xD31F: Item 1 Cantidad
0xD320: Item 2 ID
0xD321: Item 2 Cantidad
...
0xD31E + (N*2): 0xFF (terminador)
```

---

## 🔄 Flujo de Trabajo para Cambios

### Cambiar Configuración de Gimnasio

```bash
# 1. Editar configuración
code gym_scenarios/gym1_pewter_brock/team_config.json

# 2. Regenerar estado
cd gym_scenarios
python generate_gym_states.py

# 3. Probar cambio
python run_gym_comparison.py --gym 1

# 4. Verificar resultados
cat gym1_pewter_brock/results/PPO_*_report.md
```

### Agregar Nueva Métrica

```bash
# 1. Editar gym_metrics.py
code gym_scenarios/gym_metrics.py

# 2. Probar (no necesita regenerar estados)
python run_gym_comparison.py --gym 1

# 3. Verificar nueva métrica en resultados
cat gym1_pewter_brock/results/PPO_*_report.md
```

### Cambiar Paths de Modelos

```bash
# 1. Editar run_gym_comparison.py
code gym_scenarios/run_gym_comparison.py

# 2. O usar argumentos directamente
python run_gym_comparison.py --gym 1 \
    --model-base ../v2/nuevo_modelo.zip
```

---

## ⚠️ Cosas Importantes

### ⚠️ SIEMPRE regenera estados después de cambiar team_config.json

```bash
python generate_gym_states.py
```

### ⚠️ Los IDs de Pokémon son del formato interno

Ejemplo: Bulbasaur es `0x99` (153 decimal), NO 1 (Pokédex number).

Ver tabla completa en `gym_memory_addresses.py`.

### ⚠️ Stats y HP deben ser realistas

Para un Charmander nivel 12:
- HP: ~30-35
- Attack: ~15
- Defense: ~12

Usa calculadoras online o datos del juego.

### ⚠️ Cuidado con badge_bits

Es un número decimal que representa bits:
- 1 medalla = 1 (binario: 00000001)
- 2 medallas = 3 (binario: 00000011)
- 3 medallas = 7 (binario: 00000111)
- 8 medallas = 255 (binario: 11111111)

---

## 📁 Archivos que NO Debes Modificar (a menos que sepas lo que haces)

- ❌ `gym_memory_addresses.py` - Solo si agregas nuevas direcciones
- ❌ `generate_gym_states.py` - Solo si cambias lógica de generación
- ✅ `team_config.json` - **Modifica libremente**
- ⚠️ `run_gym_comparison.py` - Solo para paths de modelos
- ⚠️ `gym_metrics.py` - Solo para nuevas métricas

---

## 🎯 Ejemplos Prácticos

### Ejemplo 1: Hacer Gimnasio 1 más difícil

```json
// gym1_pewter_brock/team_config.json
{
  "player_team": [
    {
      "slot": 1,
      "species": "magikarp",      // ← Pokémon débil
      "species_id": 133,
      "level": 8,                 // ← Nivel bajo
      "moves": [{"name": "Splash", "id": 150, "pp": 40}]
    }
  ],
  "bag_items": [
    {"item": "potion", "item_id": 20, "quantity": 1}  // ← Solo 1 poción
  ]
}
```

### Ejemplo 2: Equipo sobrepoderoso para pruebas

```json
{
  "player_team": [
    {"slot": 1, "species": "mewtwo", "species_id": 131, "level": 50},
    {"slot": 2, "species": "dragonite", "species_id": 66, "level": 50},
    {"slot": 3, "species": "alakazam", "species_id": 149, "level": 50}
  ],
  "bag_items": [
    {"item": "max_potion", "item_id": 23, "quantity": 99},
    {"item": "full_restore", "item_id": 29, "quantity": 99}
  ],
  "money": 999999
}
```

### Ejemplo 3: Cambiar solo items (mantener equipo)

```json
{
  // ... equipo igual ...
  
  "bag_items": [
    {"item": "hyper_potion", "item_id": 22, "quantity": 20},
    {"item": "max_revive", "item_id": 32, "quantity": 10},
    {"item": "full_heal", "item_id": 30, "quantity": 10},
    {"item": "x_attack", "item_id": 49, "quantity": 5},
    {"item": "x_defend", "item_id": 50, "quantity": 5}
  ]
}
```

---

## ✅ Resumen de Rutas

| Qué cambiar | Archivo | Regenerar estado |
|-------------|---------|------------------|
| Equipo Pokémon | `gym{N}_{nombre}/team_config.json` | ✅ Sí |
| Items | `gym{N}_{nombre}/team_config.json` | ✅ Sí |
| Dinero | `gym{N}_{nombre}/team_config.json` | ✅ Sí |
| Medallas | `gym{N}_{nombre}/team_config.json` | ✅ Sí |
| Posición | `gym{N}_{nombre}/team_config.json` | ✅ Sí |
| Paths modelos | `run_gym_comparison.py` | ❌ No |
| Max steps | `run_gym_comparison.py` | ❌ No |
| Métricas | `gym_metrics.py` | ❌ No |
| Direcciones RAM | `gym_memory_addresses.py` | ❌ No |

---

**Para dudas, consulta README.md o los comentarios en el código** 📖
