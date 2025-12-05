# Entrega Final – Reentrenamiento (Brock Gym)

Paquete mínimo para evaluar el agente Ultimate reentrenado en el combate de Brock.

## Contenido entregado
- `run_simple.py`: ejecuta una sola batalla en Pewter Gym y genera métricas.
- `requirements.txt`: dependencias de ejecución y visualización opcional.
- `session_ultimate_38883696/ultimate_combat_final.zip`: modelo PPO final.
- `pewter_gym_configured.state`: estado inicial ubicado frente a Brock.
- `PokemonRed.gb`: ROM (no incluida; el usuario debe aportar una copia legal).
- `RESULTADOS/`: carpeta donde `run_simple.py` escribe `comparison.json`.

## Prerrequisitos
- Python 3.10+ en Windows.
- FFmpeg opcional si se desean videos (no requerido por `run_simple.py`).
- ROM de Pokémon Red con hash SHA1 `ea9bcae617fdf159b045185467ae58b2e4a48b9a` renombrada a `PokemonRed.gb` en el directorio raíz.

## Instalación rápida
1) Crear entorno (ejemplo PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2) Instalar dependencias:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```
3) Colocar `PokemonRed.gb` en la raíz del proyecto.

## Ejecutar evaluación
```powershell
python run_simple.py
```
- El script cambia internamente a `v2/`, carga `pewter_gym_configured.state` y corre el modelo `ultimate_combat_final.zip` en modo determinista.
- Se ejecuta un episodio contra Brock y se imprime una tabla con: victoria (`Win`), daño hecho/recibido, `enemy_faints`, `player_faints` y pasos.
- Los resultados se guardan en `RESULTADOS/eval_compare_<uuid>/comparison.json` con el mismo resumen del episodio.

### Campos de `comparison.json`
```json
{
  "label": "Ultimate (nuevo)",
  "win": true|false,
  "damage_dealt": float,
  "damage_received": float,
  "enemy_faints": int,
  "player_faints": int,
  "steps": int,
  "actions": [ ... ]
}
```
Si necesitas más de un episodio, edita `run_simple.py` (`step_limit` y el bucle principal) o ejecuta el script en bucle externo.

## Recompensa utilizada (CombatEvalEnv)
- Daño infligido: `+0.5` por HP; primer golpe `+5`; bonus extra al bajar al 50% (`+2`) y 20% (`+3`) de la vida rival.
- Victoria: `+20` cuando el enemigo cae a 0 HP.
- Daño recibido: `-0.1` por HP; si el jugador cae a 0 HP: `-5`.
- Menús: se favorece atacar (`+0.1` al volver al menú de ataque); penalización por navegar ítems/run (`-0.2`/`-0.5`) y anti-stall (`-0.5` tras 5 pasos de menú sin atacar).
- Salidas: abandonar combate sin dañar aplica `-5`; si se cambia al mapa 2, se aplica `-5` y se fuerza fin del episodio.
- Anti-stuck: tras 20 acciones idénticas, se fuerza una acción aleatoria básica.

## Entrenamiento y evaluación
- Modelo basado en PPO (Stable-Baselines3) entrenado en `red_gym_env_v2` con la recompensa anterior y estados de Pewter Gym.
- Evaluación determinista: usa el mismo esquema de recompensas para medir coherencia en combate.
- Recomendación: si se modifica la ROM o el estado inicial, mantener `combat_debug=True` y el límite de pasos (`step_limit=5000`) para obtener métricas comparables.

## Nota final
El agente está ajustado para derrotar a Brock desde `pewter_gym_configured.state`. Ejecuta `run_simple.py` para verificar; el JSON generado deja trazabilidad de cada corrida.
