# ✅ Checklist de Verificación - PokemonCombatAgent

## Pre-Entrenamiento

### Archivos y Dependencias
- [ ] `PokemonRed.gb` existe en `../PokemonRed.gb`
  ```powershell
  Test-Path ..\PokemonRed.gb
  ```
- [ ] `has_pokedex_nballs.state` existe en `../has_pokedex_nballs.state`
  ```powershell
  Test-Path ..\has_pokedex_nballs.state
  ```
- [ ] Todas las dependencias instaladas
  ```bash
  pip install -r requirements.txt
  ```
- [ ] Python 3.10+ instalado
  ```bash
  python --version
  ```

### Archivos del Proyecto
- [ ] `combat_gym_env.py` - Entorno principal
- [ ] `train_combat_agent.py` - Script de entrenamiento
- [ ] `compare_agents.py` - Script de comparación
- [ ] `demo_interactive.py` - Demo interactivo
- [ ] `memory_addresses.py` - Direcciones de memoria
- [ ] `requirements.txt` - Dependencias

### Documentación
- [ ] `README.md` - Leído
- [ ] `QUICKSTART.md` - Leído
- [ ] `ACTION_PLAN.md` - Revisado
- [ ] `TECHNICAL_ANALYSIS.md` - Entendido por qué TEL351 falló

---

## Prueba Inicial (5-10 minutos)

### Test Básico
- [ ] Script se ejecuta sin errores
  ```bash
  python train_combat_agent.py --timesteps 10000 --num-envs 2 --headless
  ```
- [ ] Se crea directorio `sessions/combat_session_XXXXX/`
- [ ] Ves output como:
  ```
  step: 100  victories: 0.00  hp_conserve: 5.00  W/L: 0/0
  step: 200  victories: 1.00  hp_conserve: 12.00  W/L: 1/0
  ```
- [ ] No hay errores de:
  - ❌ `FileNotFoundError` (ROM o state)
  - ❌ `AttributeError` (PyBoy API)
  - ❌ `ValueError` (dimensiones incorrectas)

---

## Entrenamiento Completo

### Antes de Lanzar
- [ ] Decidido número de CPUs (4, 8, 16)
- [ ] Verificado espacio en disco (>5GB para 1M steps)
- [ ] Cerrado otros procesos pesados
- [ ] Configurado session_name personalizado

### Durante Entrenamiento (monitorear cada 30 min)
- [ ] Reward incrementa con el tiempo
- [ ] Win Rate (W/L) mejora gradualmente
- [ ] No hay warnings/errors en consola
- [ ] Checkpoints se guardan correctamente
  ```bash
  ls sessions/combat_v1/combat_agent_*.zip
  ```

### Signos de Buen Aprendizaje
- [ ] Reward inicial: ~80-120
- [ ] Reward después 100K steps: ~150-250
- [ ] Reward después 500K steps: ~300-500
- [ ] Reward después 1M steps: >500
- [ ] Win Rate inicial: ~40-50%
- [ ] Win Rate final: >70%

### Signos de Problemas
- [ ] ❌ Reward se queda plano (<100) durante >100K steps
- [ ] ❌ Win Rate no supera 30% después de 200K steps
- [ ] ❌ Errores frecuentes en consola
- [ ] ❌ Proceso se cuelga (no avanza en minutos)

**Si hay problemas:** Ver sección Troubleshooting en QUICKSTART.md

---

## Post-Entrenamiento

### Verificación del Modelo
- [ ] Modelo final existe: `combat_agent_final.zip`
- [ ] Tamaño del archivo razonable (>50MB, <500MB)
- [ ] Stats CSV existe y tiene datos
  ```bash
  gunzip -c sessions/combat_v1/agent_stats_*.csv.gz | head
  ```
- [ ] Múltiples checkpoints guardados (100K, 200K, ..., 1M)

### Test Rápido del Modelo
- [ ] Ejecutar demo interactivo
  ```bash
  python demo_interactive.py --model sessions/combat_v1/combat_agent_final --episodes 3
  ```
- [ ] Agente juega (ves movimiento en pantalla)
- [ ] Agente gana al menos 1 de 3 batallas
- [ ] No se queda atascado en menús

---

## Comparación con Baseline

### Preparación
- [ ] Tienes modelo baseline PPO (de PokemonRedExperiments)
  ```bash
  ls ../PokemonRedExperiments/baselines/session_*/poke_*.zip
  ```
- [ ] O has entrenado tu propio baseline

### Ejecución
- [ ] Script de comparación corre sin errores
  ```bash
  python compare_agents.py --combat-agent MODEL1 --baseline-agent MODEL2 --episodes 20
  ```
- [ ] Se crean archivos en `comparison_results/`
  - [ ] `combat_agent_metrics.csv`
  - [ ] `baseline_agent_metrics.csv`
  - [ ] `comparison_results.csv`
  - [ ] `summary.json`

### Resultados Esperados
- [ ] Combat Agent Win Rate > Baseline Win Rate (+10-25%)
- [ ] Combat Agent HP Conservation > Baseline (+15-30%)
- [ ] p-value < 0.05 en al menos 2 métricas principales
- [ ] Cohen's d > 0.5 (efecto mediano o grande)

---

## Análisis Cualitativo

### Observaciones del Agente
- [ ] Usa curación cuando HP < 50% (no cuando HP > 80%)
- [ ] Cambia Pokémon cuando tiene desventaja clara
- [ ] No se queda atascado en loops (menú, misma acción repetida)
- [ ] Progresa en el juego (gana batallas, explora)

### Videos/Screenshots
- [ ] Captura de pantalla de victoria eficiente (HP > 70% al final)
- [ ] Captura de batalla perdida (para análisis)
- [ ] (Opcional) Video de 1-2 minutos mostrando comportamiento

---

## Documentación Final

### Reporte Técnico
- [ ] Introducción: Problema y motivación
- [ ] Metodología: Arquitectura, recompensas, configuración
- [ ] Resultados: Tablas comparativas con p-values
- [ ] Gráficos: Win rate, HP conservation, learning curves
- [ ] Análisis: Por qué combat agent es mejor
- [ ] Conclusiones: Resultados validados estadísticamente
- [ ] Limitaciones: Qué no funciona perfectamente
- [ ] Trabajo Futuro: Extensiones posibles

### Datos de Soporte
- [ ] `comparison_results/` completo
- [ ] Gráficos exportados (PNG/PDF)
- [ ] Checkpoints compartibles (ZIP del mejor modelo)
- [ ] Agent stats CSV (raw data)

---

## Entregables Finales

### Código
- [ ] Repositorio `PokemonCombatAgent/` limpio
- [ ] Sin archivos temporales (*.pyc, __pycache__)
- [ ] README.md actualizado con resultados reales
- [ ] Comentarios en código crítico

### Datos
- [ ] Modelo entrenado (combat_agent_final.zip)
- [ ] Métricas de evaluación (CSVs)
- [ ] Comparación estadística (comparison_results.csv)

### Documentación
- [ ] Reporte técnico (PDF)
- [ ] Presentación (PPT/PPTX) si es necesario
- [ ] README con instrucciones de reproducción

### Opcional (Para publicación/tesis)
- [ ] Paper académico (formato IEEE/ACM)
- [ ] Video demo (YouTube/Drive)
- [ ] GitHub repo público
- [ ] Poster científico

---

## Validación Científica

### Hipótesis
- [ ] H0 y H1 claramente definidas
- [ ] Test estadístico elegido (t-test pareado)
- [ ] Nivel de significancia (α = 0.05)
- [ ] Potencia estadística adecuada (>0.80)

### Resultados
- [ ] p-value calculado y reportado
- [ ] Cohen's d (effect size) calculado
- [ ] Intervalos de confianza (95% CI)
- [ ] Conclusión estadística justificada

### Reproducibilidad
- [ ] Seeds documentados
- [ ] Hiperparámetros registrados
- [ ] Versiones de librerías especificadas
- [ ] Estados iniciales compartidos

---

## Control de Calidad

### Código
- [ ] Sin errores de sintaxis
- [ ] Sin warnings importantes
- [ ] Funciones documentadas
- [ ] Variables con nombres descriptivos

### Datos
- [ ] Sin valores faltantes (NaN) inesperados
- [ ] Rango de valores razonable (0-100% para HP, etc.)
- [ ] Suficientes muestras (100+ episodios)
- [ ] Distribuciones no sesgadas

### Documentación
- [ ] Sin errores de ortografía
- [ ] Gráficos con etiquetas claras
- [ ] Tablas bien formateadas
- [ ] Referencias completas

---

## Pre-Presentación/Entrega

### Revisión Final
- [ ] Todos los archivos en el lugar correcto
- [ ] README.md actualizado con resultados finales
- [ ] Números verificados (no copiar sin revisar)
- [ ] Gráficos de alta calidad (300 DPI para paper)

### Backup
- [ ] Código respaldado (GitHub, Drive, etc.)
- [ ] Datos importantes respaldados
- [ ] Checkpoints de modelos guardados
- [ ] Documentación en múltiples formatos (MD, PDF)

### Última Verificación
- [ ] Ejecutar training desde cero funciona
- [ ] Comparación genera resultados esperados
- [ ] Demo interactivo muestra comportamiento correcto
- [ ] Reporte está completo y coherente

---

## 🎉 ¡Listo para Entregar!

Si todos los checks están ✅, tienes un proyecto completo, validado y reproducible.

**Siguientes pasos opcionales:**
- Publicar en GitHub
- Crear video explicativo
- Escribir blog post
- Presentar en conferencia
- Extender a otros agentes (puzzle, exploration)

---

**Fecha de Verificación:** _____________

**Verificado por:** _____________

**Notas adicionales:**
```
___________________________________________________________
___________________________________________________________
___________________________________________________________
```
