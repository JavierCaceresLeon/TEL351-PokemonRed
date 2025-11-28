Write-Host "Intentando cerrar procesos de Python para liberar archivos (puede cerrar tu kernel actual)..."
Stop-Process -Name "python" -ErrorAction SilentlyContinue

Write-Host "Desinstalando Torch..."
& "C:\Users\javi1\anaconda3\envs\pokeenv\python.exe" -m pip uninstall -y torch torchvision torchaudio

Write-Host "Limpiando caché..."
& "C:\Users\javi1\anaconda3\envs\pokeenv\python.exe" -m pip cache purge

Write-Host "Instalando Torch 2.4.1 (Versión Estable)..."
& "C:\Users\javi1\anaconda3\envs\pokeenv\python.exe" -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

Write-Host "Verificando instalación..."
& "C:\Users\javi1\anaconda3\envs\pokeenv\python.exe" -c "import torch; print('Torch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"

Write-Host "¡Listo! Ahora reinicia el Kernel en VS Code y vuelve a ejecutar el notebook."