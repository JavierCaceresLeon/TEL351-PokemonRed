# 🎮 Pokemon Combat Agent - Quick Start

## ✅ Setup Completado

El proyecto está listo para entrenar. Todos los archivos necesarios están en su lugar:

- ✅ `PokemonRed.gb` - ROM de Pokemon Red
- ✅ `has_pokedex_nballs.state` - Estado inicial con Pokedex y Pokeballs
- ✅ `combat_gym_env.py` - Entorno especializado en combate
- ✅ `train_combat_agent.py` - Script de entrenamiento
- ✅ Dependencias instaladas (PyBoy 2.6+, Stable-Baselines3, etc.)

## 🚀 Iniciar Entrenamiento

### Opción 1: Script PowerShell (Recomendado para Windows)

```powershell
cd PokemonCombatAgent
.\start_training.ps1
```

### Opción 2: Comando directo

```powershell
cd PokemonCombatAgent
python train_combat_agent.py --timesteps 1000000 --num-envs 4 --headless
```

### Opción 3: Entrenamiento rápido de prueba (100K pasos)

```powershell
python train_combat_agent.py --timesteps 100000 --num-envs 2 --headless
```

## 📊 Monitoreo del Entrenamiento

### TensorBoard (Recomendado)

Mientras el entrenamiento corre, abre otra terminal:

```powershell
cd PokemonCombatAgent/sessions
tensorboard --logdir .
```

Luego abre tu navegador en: `http://localhost:6006`

### Archivos generados

- `sessions/combat_session_XXXXX/` - Directorio de la sesión
- `sessions/combat_session_XXXXX/combat_agent_XXXXX_steps.zip` - Checkpoints cada 100K pasos
- `sessions/combat_session_XXXXX/combat_agent_final.zip` - Modelo final

## ⚙️ Parámetros de Entrenamiento

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--timesteps` | 1,000,000 | Pasos totales de entrenamiento |
| `--num-envs` | 4 | Entornos paralelos (4 óptimo para Windows) |
| `--headless` | - | Modo sin ventana gráfica |
| `--learning-rate` | 0.0001 | Tasa de aprendizaje de PPO |
| `--checkpoint-freq` | 100,000 | Frecuencia de guardado |

### Ejemplo con parámetros personalizados:

```powershell
python train_combat_agent.py `
    --timesteps 3000000 `
    --num-envs 8 `
    --learning-rate 0.00005 `
    --headless
```

## 🎯 Recompensas de Combate

El agente está optimizado para:

- ✅ **Victorias en batalla** (+1000 puntos)
- ✅ **Eficiencia en combate** (bonus por conservar HP)
- ✅ **Ventaja de tipos** (bonus por usar movimientos efectivos)
- ✅ **Enfrentar oponentes fuertes** (bonus por nivel del oponente)
- ❌ **Penalizaciones** por perder batallas o curarse innecesariamente

## 📈 Tiempo Estimado de Entrenamiento

- **100K pasos** (~10-15 minutos) - Prueba rápida
- **1M pasos** (~2-3 horas) - Entrenamiento básico
- **3M pasos** (~6-9 horas) - Entrenamiento completo recomendado

*Tiempos aproximados con 4 entornos en CPU moderna*

## 🔧 Solución de Problemas

### Error: "ROM file not found"
```powershell
# Verifica que el ROM esté en la carpeta correcta
Test-Path ./PokemonRed.gb
```

### Error: "State file not found"
```powershell
# Verifica el archivo de estado
Test-Path ./has_pokedex_nballs.state
```

### Entrenamiento muy lento
- Reduce `--num-envs` a 2
- Verifica que `--headless` esté activado
- Cierra programas innecesarios

### Memoria insuficiente
- Reduce `--num-envs` a 2 o 1
- Reduce `--batch-size` (default: 512)

## 📝 Siguientes Pasos

1. **Entrenar modelo de combate**: `python train_combat_agent.py`
2. **Obtener baseline**: Copiar desde `PokemonRedExperiments` o entrenar uno nuevo
3. **Comparar modelos**: `python compare_agents.py --combat-model sessions/.../final.zip --baseline-model path/to/baseline.zip`
4. **Análisis interactivo**: `python demo_interactive.py --model sessions/.../final.zip`

## 📚 Documentación Adicional

- `ACTION_PLAN.md` - Plan de 5 días para entrenamiento completo
- `TECHNICAL_ANALYSIS.md` - Análisis de por qué TEL351 falló
- `EXECUTIVE_SUMMARY.md` - Resumen ejecutivo del proyecto
- `CHECKLIST.md` - Lista de verificación de setup

## 🆘 Soporte

Si encuentras problemas:
1. Revisa `TECHNICAL_ANALYSIS.md` para errores comunes
2. Verifica que todas las dependencias estén instaladas: `pip install -r requirements.txt`
3. Comprueba la versión de Python: `python --version` (requiere 3.10+)

---

**¡Listo para entrenar! 🚀**
