# Tabu Search Algorithm

Esta carpeta contiene la implementación del algoritmo **Tabu Search** para Pokemon Red.

## Archivos

- `tabu_agent.py` - Implementación del algoritmo Tabu Search con anti-spam
- `demo_tabu_search.py` - Script de demostración del Tabu Search

## Estado Actual

 **PROBLEMAS CONOCIDOS**: 
- El Tabu Search "funciona asqueros, se pega en el menu"
- A pesar de las mejoras anti-spam, sigue teniendo problemas de performance
- Se recomienda usar **epsilon greedy** o **A*** en su lugar

## Características Implementadas

- **Lista Tabú**: Evita revisitar estados recientes
- **Anti-menu spam**: Detecta y penaliza abuse del menú
- **Cooldowns**: Previene acciones repetitivas
- **Detección de ciclos**: Identifica patrones repetitivos
- **Métricas anti-spam**: Tracking de comportamientos problemáticos

## Problemas Identificados

1. **Menu Spam**: El agente se queda atascado abriendo/cerrando menús
2. **Ciclos**: A pesar de la lista tabú, puede entrar en bucles
3. **Performance Pobre**: No converge efectivamente hacia objetivos
4. **Exploración Ineficiente**: Explora de manera caótica sin dirección

## Mejoras Intentadas

- Detección de abuso de menú con `detect_menu_abuse()`
- Penalización de acciones de menú con `calculate_menu_penalty()`
- Cooldowns para prevenir spam
- Lista tabú mejorada con decay temporal
- **Resultado**: Aún funciona mal comparado con epsilon greedy

## Uso (No Recomendado)

```bash
python tabu_search/demo_tabu_search.py
```

## Recomendación

**NO usar Tabu Search**. Los algoritmos alternativos funcionan mucho mejor:

- **Epsilon Greedy**: Funciona de maravilla, muy recomendado
- **A***: Excelente para pathfinding dirigido por objetivos

## Lecciones Aprendidas

- La lista tabú no es suficiente para evitar comportamientos problemáticos en Pokemon Red
- El menú del juego crea problemas únicos que son difíciles de resolver con Tabu Search
- Algoritmos más simples (epsilon greedy) pueden superar métodos más complejos