import json, shutil
from pathlib import Path

project = Path("C:/Users/javi1/Documents/repos_git/TEL351-PokemonRed")
scenarios = json.loads((project/"gym_scenarios/scenarios.json").read_text())

for scenario in scenarios["scenarios"]:
    folder = next((p for p in (project/"gym_scenarios").glob(f"gym*_{scenario['id']}")), None)
    if not folder:
        print(f"[!] Carpeta para {scenario['id']} no encontrada")
        continue

    generated = folder / "gym_scenario.state"
    if not generated.exists():
        print(f"[!] {generated} no existe (ejecuta generate_gym_states.py).")
        continue

    for phase in scenario["phases"]:
        target = project / phase["state_file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated, target)
        print(f"[✓] Copiado {generated.name} -> {target}")