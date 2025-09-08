# GUÍA DE CONFIGURACIÓN E INSTRUCCIONES

## ¡Proyecto Completado! 🎉

Se ha creado exitosamente un sistema de comparación entre el agente entrenado (v2) y algoritmos de búsqueda clásicos (A*, Tabú Search) para salir de la habitación inicial en Pokémon Red.

## Estructura Creada

```
TEL351-PokemonRed/
├── v2/                           # (Existente) Modelo entrenado
└── comparison_agents/            # (NUEVO) Sistema de comparación
    ├── README.md                 # Documentación completa
    ├── config.py                 # Configuración centralizada
    ├── requirements.txt          # Dependencias necesarias
    ├── install_dependencies.py   # Script de instalación
    ├── test_setup.py            # Script de verificación
    ├── run_comparison.py        # Script principal
    ├── search_env.py            # Entorno para algoritmos de búsqueda
    ├── v2_agent.py              # Wrapper para modelo v2
    ├── search_algorithms/
    │   ├── astar_agent.py       # Implementación A*
    │   └── tabu_agent.py        # Implementación Tabú Search
    └── results/                 # Resultados de comparaciones
```

## Pasos para Usar el Sistema

### 1. Instalar Dependencias

```bash
cd comparison_agents

# Opción A: Script automático
python install_dependencies.py

# Opción B: Manual
pip install -r requirements.txt
```

### 2. Verificar Configuración

```bash
python test_setup.py
```

Este script verifica que:
- Todos los módulos se importen correctamente
- Los archivos requeridos (PokemonRed.gb, init.state) existen
- Los agentes funcionan básicamente

### 3. Ejecutar Comparación

```bash
# Comparación completa
python run_comparison.py

# O editar config.py para ajustar parámetros
```

## Características del Sistema

### ✅ Agentes Implementados

1. **Agente V2** (Modelo Entrenado)
   - Usa el modelo PPO entrenado en `/v2`
   - Predicciones rápidas y consistentes
   - Requiere modelo pre-entrenado

2. **Agente A***
   - Búsqueda informada con heurística
   - Encuentra caminos óptimos
   - Configurable (profundidad, heurística)

3. **Agente Tabú Search**
   - Búsqueda local con memoria
   - Evita óptimos locales
   - Parámetros ajustables (iteraciones, tamaño tabú)

### ✅ Métricas de Comparación

- **Tasa de éxito**: ¿Logra salir de la habitación?
- **Tiempo total**: Desde inicio hasta completar objetivo
- **Número de pasos**: Acciones tomadas
- **Eficiencia**: Pasos por segundo
- **Métricas específicas**: Nodos explorados (A*), iteraciones (Tabú)

### ✅ Configuración Flexible

- Parámetros centralizados en `config.py`
- Configuraciones predefinidas (QuickTest, Intensive, Debug)
- Fácil ajuste de límites y parámetros

### ✅ Preservación del Proyecto Original

- **No altera nada del v2 existente**
- Directorio separado `comparison_agents/`
- Importa dependencias sin modificar estructura original

## Resultados Esperados

### Agente V2 (Entrenado)
- ✅ **Ventajas**: Rápido, consistente, pre-optimizado
- ⚠️ **Desventajas**: Caja negra, requiere modelo entrenado

### A*
- ✅ **Ventajas**: Óptimo, explícito, comprensible
- ⚠️ **Desventajas**: Puede ser lento, requiere buena heurística

### Tabú Search
- ✅ **Ventajas**: Bueno para espacios grandes, evita óptimos locales
- ⚠️ **Desventajas**: No garantiza optimalidad, variable

## Archivos de Resultados

Los resultados se guardan automáticamente en:
```
results/comparison_results_YYYYMMDD_HHMMSS.json
```

Formato incluye:
- Tiempos de ejecución
- Número de pasos
- Tasas de éxito
- Métricas específicas por agente
- Timestamps para seguimiento

## Próximos Pasos Sugeridos

### Inmediatos
1. Instalar dependencias
2. Ejecutar `test_setup.py`
3. Ejecutar `run_comparison.py` con configuración básica
4. Analizar primeros resultados

### Mejoras Futuras
1. **Heurísticas más sofisticadas** para A*
2. **Parámetros adaptativos** para Tabú Search
3. **Más algoritmos**: Beam Search, Genetic Algorithm
4. **Visualización**: Trayectorias, mapas de calor
5. **Análisis estadístico**: Intervalos de confianza, pruebas de significancia

## Solución de Problemas

### Dependencias Faltantes
```bash
pip install pyboy stable-baselines3 numpy scikit-image matplotlib
```

### Modelo V2 No Encontrado
- Verificar que existe un archivo `.zip` en `v2/runs/`
- Entrenar modelo v2 si no existe

### ROM o Estado No Encontrado
- Verificar rutas en `config.py`
- Asegurar que `PokemonRed.gb` e `init.state` existen

## Contacto y Contribuciones

Para mejoras o problemas:
1. Revisar documentación en `README.md`
2. Ejecutar `test_setup.py` para diagnóstico
3. Verificar configuración en `config.py`

¡El sistema está listo para comparar agentes! 🚀
