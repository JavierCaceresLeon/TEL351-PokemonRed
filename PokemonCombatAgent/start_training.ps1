# Pokemon Combat Agent - Training Script
# Executes training with optimal settings for Windows

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "    Pokemon Combat Agent - Training Session" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment if it exists
$venvPath = "..\\.venv\\Scripts\\python.exe"
if (Test-Path $venvPath) {
    Write-Host "Using virtual environment Python..." -ForegroundColor Green
    $pythonCmd = $venvPath
} else {
    Write-Host "Using system Python..." -ForegroundColor Yellow
    $pythonCmd = "python"
}

# Default parameters (can be overridden)
$timesteps = 1000000
$numEnvs = 4

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  - Total timesteps: $timesteps" -ForegroundColor White
Write-Host "  - Parallel environments: $numEnvs" -ForegroundColor White
Write-Host "  - ROM: ./PokemonRed.gb" -ForegroundColor White
Write-Host "  - Initial state: ./has_pokedex_nballs.state" -ForegroundColor White
Write-Host ""
Write-Host "Training will start in 3 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Run training
& $pythonCmd train_combat_agent.py `
    --timesteps $timesteps `
    --num-envs $numEnvs `
    --headless

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "Training session completed!" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
