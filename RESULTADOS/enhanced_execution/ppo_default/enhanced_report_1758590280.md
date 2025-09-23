# Reporte de Ejecución Profesional: PPO

## Configuración
- **Algoritmo:** ppo
- **Variante:** default
- **Timestamp:** 1758590280
- **Fecha:** 2025-09-22 22:18:00

## Resultados de Rendimiento

### Métricas Principales
- **Estado de Ejecución:** ❌ Fallido
- **Tiempo de Ejecución:** 10.96 segundos
- **Pasos Totales:** 0
- **Recompensa Total:** 0.00
- **Eficiencia:** 0.0000 recompensa/paso
- **Velocidad:** 0.0 pasos/segundo

### Análisis de Rendimiento

La ejecución falló con el siguiente error: You have passed a tuple to the predict() function instead of a Numpy array or a Dict. You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) vs `obs = vec_env.reset()` (SB3 VecEnv). See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api

**Análisis del Fallo:**
- Tiempo transcurrido antes del fallo: 10.96 segundos
- Posibles causas: Error de implementación, falta de dependencias, o configuración incorrecta


### Historial de Métricas
Se recolectaron 0 puntos de datos durante la ejecución.

### Estado Final
{}

---
*Reporte generado automáticamente por el Sistema de Ejecución Profesional*
