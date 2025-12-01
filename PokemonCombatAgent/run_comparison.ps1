"""
Quick comparison script - Run this to compare models
"""

# Combat Agent (your trained model)
$combat_model = "sessions\combat_agent_final\combat_agent_final.zip"

# Baseline from PokemonRedExperiments
$baseline_model = "..\PokemonRedExperiments\v2\runs\poke_26214400.zip"

# Choose a battle state (or use has_pokedex_nballs.state for early game)
$battle_state = "has_pokedex_nballs.state"

# Number of episodes to compare
$episodes = 3

Write-Host "Starting Model Comparison..." -ForegroundColor Green
Write-Host "Combat Model: $combat_model" -ForegroundColor Cyan
Write-Host "Baseline Model: $baseline_model" -ForegroundColor Cyan
Write-Host "Battle State: $battle_state" -ForegroundColor Cyan
Write-Host ""

python compare_models_interactive.py `
    --combat-model $combat_model `
    --baseline-model $baseline_model `
    --battle-state $battle_state `
    --episodes $episodes
