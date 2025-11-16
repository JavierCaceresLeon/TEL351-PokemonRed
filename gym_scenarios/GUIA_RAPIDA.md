# 🚀 GUÍA RÁPIDA DE USO

## 📦 Lo que se ha creado

```
gym_scenarios/
│
├── 📄 README.md                    ← Documentación completa
├── 📄 RESUMEN_EJECUTIVO.md         ← Resumen del sistema
├── 📄 GUIA_RAPIDA.md               ← Esta guía
│
├── 🐍 gym_memory_addresses.py      ← Direcciones RAM Pokémon
├── 🐍 gym_metrics.py               ← Sistema de métricas  
├── 🐍 generate_gym_states.py       ← Generador de .state
├── 🐍 run_gym_comparison.py        ← Script principal
│
└── 🏟️ gym1-8_*/                    ← 8 carpetas de gimnasios
    ├── team_config.json            ← Equipo configurado
    └── gym_scenario.state          ← (se genera)
```

---

## ⚡ 3 Pasos para Empezar

### 1️⃣ Generar Estados

```bash
cd gym_scenarios
python generate_gym_states.py
```

**Resultado**: 8 archivos `.state` creados

### 2️⃣ Evaluar un Gimnasio

```bash
python run_gym_comparison.py --gym 1
```

**Resultado**: Reportes en `gym1_pewter_brock/results/`

### 3️⃣ Ver Resultados

```bash
cd gym1_pewter_brock/results/
cat PPO_Base_gym1_*_report.md
cat comparison_*.json
```

---

## 📊 Comandos Útiles

### Evaluar un gimnasio específico
```bash
python run_gym_comparison.py --gym 1    # Brock
python run_gym_comparison.py --gym 3    # Lt. Surge
python run_gym_comparison.py --gym 6    # Sabrina
```

### Evaluar todos los gimnasios
```bash
python run_gym_comparison.py --all
```

### Usar modelos personalizados
```bash
python run_gym_comparison.py --gym 1 \
    --model-base ../v2/mi_modelo_base.zip \
    --model-retrained ../v2/mi_modelo_mejorado.zip
```

### Más pasos (para gimnasios difíciles)
```bash
python run_gym_comparison.py --gym 6 --max-steps 20000
```

---

## 🔧 Modificar Equipos

### 1. Editar configuración
```bash
code gym1_pewter_brock/team_config.json
```

### 2. Cambiar lo que necesites
```json
{
  "player_team": [
    {
      "species": "pikachu",      // ← Cambiar Pokémon
      "level": 15,               // ← Cambiar nivel
      "moves": [...]             // ← Cambiar movimientos
    }
  ],
  "bag_items": [
    {"item": "hyper_potion", "quantity": 10}  // ← Más items
  ],
  "money": 10000               // ← Más dinero
}
```

### 3. Regenerar estado
```bash
python generate_gym_states.py
```

---

## 📈 Leer Resultados

### Archivo Markdown (más legible)
```bash
cat gym1_pewter_brock/results/PPO_Base_gym1_*_report.md
```

Contiene:
- ✅/❌ Éxito o fallo
- ⏱️ Tiempo y pasos
- 🎯 Recompensas
- ⚔️ Resultado de batalla
- 🧩 Estado del puzzle
- 📊 Distribución de acciones

### Archivo JSON (para análisis)
```python
import json

with open('gym1_pewter_brock/results/comparison_*.json') as f:
    data = json.load(f)
    
# Ver diferencias
for metric, diff in data['differences'].items():
    print(f"{metric}: {diff['percent_change']:.1f}% change")
```

### Archivo CSV (para Excel)
```bash
# Abrir en Excel/LibreOffice
gym1_pewter_brock/results/PPO_Base_gym1_*_summary.csv
```

---

## 🏆 Tabla de Gimnasios

| # | Comando | Gimnasio | Tipo | Puzzle |
|---|---------|----------|------|--------|
| 1 | `--gym 1` | Brock (Pewter) | Roca | ❌ |
| 2 | `--gym 2` | Misty (Cerulean) | Agua | ❌ |
| 3 | `--gym 3` | Lt. Surge (Vermilion) | Eléctrico | ✅ Botes |
| 4 | `--gym 4` | Erika (Celadon) | Planta | ❌ |
| 5 | `--gym 5` | Koga (Fuchsia) | Veneno | ✅ Paredes |
| 6 | `--gym 6` | Sabrina (Saffron) | Psíquico | ✅ Teleports |
| 7 | `--gym 7` | Blaine (Cinnabar) | Fuego | ✅ Quiz |
| 8 | `--gym 8` | Giovanni (Viridian) | Tierra | ✅ Trainers |

---

## 🎮 Equipos Configurados

### Gimnasio 1: Brock (Fácil)
- Charmander Lv.12 (débil a Roca)
- Pidgey Lv.9
- Rattata Lv.8

### Gimnasio 3: Lt. Surge (Medio)
- Wartortle Lv.25 (débil a Eléctrico)
- Pidgeotto Lv.23
- Diglett Lv.22 (clave: Tierra)
- Nidorino Lv.23

### Gimnasio 6: Sabrina (Difícil)
- Charizard Lv.42
- Pidgeot Lv.40
- Rhydon Lv.40
- Gyarados Lv.39 (Bite anti-psíquico)
- Electabuzz Lv.39
- Hitmonlee Lv.38

### Gimnasio 8: Giovanni (Muy Difícil)
- Blastoise Lv.50 (débil a Tierra)
- Pidgeot Lv.48
- Venusaur Lv.49
- Arcanine Lv.48
- Alakazam Lv.48
- Lapras Lv.47

---

## 🔍 Direcciones de Memoria Clave

### Equipo
```python
PARTY_SIZE = 0xD163          # Tamaño del equipo
PARTY_POKEMON = 0xD164-0xD169  # Especies
LEVELS = 0xD18C, 0xD1B8, ...   # Niveles
HP = 0xD16C, 0xD198, ...       # HP actual
```

### Items
```python
BAG_ITEMS = 0xD31E           # Inicio de items
# Format: [item_id, quantity, item_id, quantity, ...]
```

### Posición
```python
X_POS = 0xD362
Y_POS = 0xD361
MAP_ID = 0xD35E
```

### Medallas
```python
BADGES = 0xD356              # Byte con bits de medallas
# Bits: 0=Boulder, 1=Cascade, 2=Thunder, ...
```

Ver `gym_memory_addresses.py` para la lista completa.

---

## 🎯 Métricas Capturadas

- ⏱️ **Tiempo**: Segundos, pasos, velocidad
- 🎁 **Recompensa**: Total, promedio, máx, mín  
- ⚔️ **Combate**: Victoria, pasos de batalla
- 🧩 **Puzzle**: Resuelto, intentos
- 🗺️ **Navegación**: Exploración, atascamientos
- 🎒 **Items**: Pociones, curas usadas
- 💪 **Equipo**: HP promedio final

---

## 🆘 Solución de Problemas

### Error: ROM no encontrado
```bash
# Solución: Coloca PokemonRed.gb en la raíz
cp /path/to/PokemonRed.gb ../
```

### Error: Estado no generado
```bash
# Solución: Ejecuta el generador primero
python generate_gym_states.py
```

### Error: Modelo no encontrado
```bash
# Solución: Especifica el path correcto
python run_gym_comparison.py --gym 1 \
    --model-base ../v2/tu_modelo.zip
```

### Ver opciones disponibles
```bash
python run_gym_comparison.py --help
```

---

## 📚 Archivos para Leer

1. **Esta guía** - Inicio rápido ⚡
2. **RESUMEN_EJECUTIVO.md** - Qué se creó y por qué 📊
3. **README.md** - Documentación completa 📖
4. **gym_memory_addresses.py** - Referencia RAM 🗺️

---

## ✅ Checklist

- [ ] ROM `PokemonRed.gb` en raíz
- [ ] `cd gym_scenarios`
- [ ] `python generate_gym_states.py`
- [ ] Verificar archivos `.state` creados
- [ ] `python run_gym_comparison.py --gym 1`
- [ ] Revisar resultados en `results/`
- [ ] (Opcional) `python run_gym_comparison.py --all`

---

## 🎉 ¡Listo!

El sistema está **100% funcional**. Solo ejecuta:

```bash
cd gym_scenarios
python generate_gym_states.py
python run_gym_comparison.py --all
```

Y tendrás comparaciones completas de PPO Base vs Reentrenado en los 8 gimnasios.

---

**Para más detalles, consulta README.md** 📖
