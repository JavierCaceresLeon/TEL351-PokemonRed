"""
Sistema de Métricas para Escenarios de Gimnasios
================================================

Sistema completo de medición de desempeño para agentes PPO en gimnasios.
Mide: tiempo, pasos, éxito en puzzles, éxito en combates, uso de items, etc.

Uso:
    from gym_metrics import GymMetricsTracker
    
    tracker = GymMetricsTracker(gym_number=1, agent_name="PPO_Base")
    tracker.start()
    # ... ejecutar agente ...
    tracker.record_step(action, reward, game_state)
    tracker.end(success=True)
    tracker.save_metrics()
"""

import time
import json
import csv
from pathlib import Path
from datetime import datetime
import numpy as np


class GymMetricsTracker:
    """Rastrea métricas detalladas para evaluación de gimnasios"""
    
    def __init__(self, gym_number, agent_name, gym_name="", session_id=None):
        """
        Inicializa el tracker de métricas
        
        Args:
            gym_number: Número del gimnasio (1-8)
            agent_name: Nombre del agente ("PPO_Base", "PPO_Retrained", etc.)
            gym_name: Nombre descriptivo del gimnasio
            session_id: ID único de sesión (opcional)
        """
        self.gym_number = gym_number
        self.agent_name = agent_name
        self.gym_name = gym_name
        self.session_id = session_id or f"{int(time.time())}"
        
        # Métricas de tiempo
        self.start_time = None
        self.end_time = None
        self.total_duration = 0
        
        # Métricas de pasos
        self.total_steps = 0
        self.step_history = []
        
        # Métricas de recompensa
        self.total_reward = 0
        self.reward_history = []
        self.max_reward = -float('inf')
        self.min_reward = float('inf')
        
        # Métricas de combate
        self.battle_started = False
        self.battle_won = False
        self.battle_steps = 0
        self.damage_dealt = 0
        self.damage_received = 0
        self.pokemon_fainted_player = 0
        self.pokemon_fainted_opponent = 0
        
        # Métricas de puzzle
        self.puzzle_attempts = 0
        self.puzzle_solved = False
        self.puzzle_steps = 0
        
        # Métricas de navegación
        self.positions_visited = set()
        self.unique_tiles_explored = 0
        self.backtrack_count = 0
        self.stuck_count = 0
        self.last_position = None
        self.stuck_threshold = 50  # pasos sin movimiento
        self.stuck_counter = 0
        
        # Métricas de items
        self.items_used = []
        self.potions_used = 0
        self.status_heals_used = 0
        
        # Métricas de equipo
        self.initial_team_hp = []
        self.final_team_hp = []
        self.team_levels = []
        
        # Estado del juego
        self.game_states = []
        self.action_distribution = {}
        
        # Resultado final
        self.success = False
        self.failure_reason = ""
        
        # Metadata
        self.metadata = {
            'gym_number': gym_number,
            'gym_name': gym_name,
            'agent_name': agent_name,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat()
        }
    
    def start(self):
        """Inicia el tracking de métricas"""
        self.start_time = time.time()
        print(f"📊 Métricas iniciadas para {self.agent_name} en Gimnasio {self.gym_number}")
    
    def record_step(self, action, reward, game_state):
        """
        Registra un paso del agente
        
        Args:
            action: Acción tomada
            reward: Recompensa obtenida
            game_state: Diccionario con estado del juego (HP, posición, etc.)
        """
        self.total_steps += 1
        
        # Recompensa
        self.total_reward += reward
        self.reward_history.append(reward)
        self.max_reward = max(self.max_reward, reward)
        self.min_reward = min(self.min_reward, reward)
        
        # Acción
        if action not in self.action_distribution:
            self.action_distribution[action] = 0
        self.action_distribution[action] += 1
        
        # Posición
        position = (game_state.get('x', 0), game_state.get('y', 0))
        if position not in self.positions_visited:
            self.positions_visited.add(position)
            self.unique_tiles_explored += 1
        
        # Detectar stuck (atascado)
        if position == self.last_position:
            self.stuck_counter += 1
            if self.stuck_counter >= self.stuck_threshold:
                self.stuck_count += 1
                self.stuck_counter = 0
        else:
            self.stuck_counter = 0
        
        self.last_position = position
        
        # Guardar snapshot cada 100 pasos
        if self.total_steps % 100 == 0:
            self.game_states.append({
                'step': self.total_steps,
                'position': position,
                'reward': reward,
                'hp': game_state.get('hp', []),
                'in_battle': game_state.get('in_battle', False)
            })
    
    def record_battle_start(self):
        """Registra el inicio de un combate"""
        self.battle_started = True
        self.battle_steps = 0
        print(f"  ⚔️ Batalla iniciada en paso {self.total_steps}")
    
    def record_battle_end(self, won=False):
        """Registra el fin de un combate"""
        self.battle_won = won
        result = "VICTORIA" if won else "DERROTA"
        print(f"  🏆 Batalla terminada: {result} (duración: {self.battle_steps} pasos)")
    
    def record_puzzle_attempt(self):
        """Registra un intento de resolver el puzzle"""
        self.puzzle_attempts += 1
    
    def record_puzzle_solved(self):
        """Registra que el puzzle fue resuelto"""
        self.puzzle_solved = True
        self.puzzle_steps = self.total_steps
        print(f"  🧩 Puzzle resuelto en {self.puzzle_steps} pasos")
    
    def record_item_used(self, item_name):
        """Registra el uso de un item"""
        self.items_used.append({
            'step': self.total_steps,
            'item': item_name
        })
        
        if 'potion' in item_name.lower():
            self.potions_used += 1
        elif 'heal' in item_name.lower() or 'antidote' in item_name.lower():
            self.status_heals_used += 1
    
    def record_team_state(self, team_hp, is_final=False):
        """
        Registra el estado del equipo
        
        Args:
            team_hp: Lista de HP de cada Pokémon
            is_final: Si es el estado final
        """
        if not self.initial_team_hp:
            self.initial_team_hp = team_hp.copy()
        
        if is_final:
            self.final_team_hp = team_hp.copy()
    
    def end(self, success=False, failure_reason=""):
        """
        Finaliza el tracking de métricas
        
        Args:
            success: Si el gimnasio fue completado exitosamente
            failure_reason: Razón del fallo (si aplica)
        """
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time
        self.success = success
        self.failure_reason = failure_reason
        
        result = "✓ ÉXITO" if success else "✗ FALLO"
        print(f"\n📊 Métricas finalizadas: {result}")
        print(f"   Duración: {self.total_duration:.2f}s")
        print(f"   Pasos: {self.total_steps}")
        print(f"   Recompensa total: {self.total_reward:.2f}")
    
    def get_summary_stats(self):
        """Retorna estadísticas resumidas"""
        avg_reward = self.total_reward / max(self.total_steps, 1)
        steps_per_second = self.total_steps / max(self.total_duration, 1)
        
        # Calcular HP final promedio
        final_hp_avg = 0
        if self.final_team_hp:
            final_hp_avg = np.mean(self.final_team_hp)
        
        return {
            # Rendimiento general
            'success': self.success,
            'failure_reason': self.failure_reason,
            'total_duration_seconds': round(self.total_duration, 2),
            'total_steps': self.total_steps,
            'steps_per_second': round(steps_per_second, 2),
            
            # Recompensas
            'total_reward': round(self.total_reward, 2),
            'avg_reward_per_step': round(avg_reward, 4),
            'max_reward': round(self.max_reward, 2),
            'min_reward': round(self.min_reward, 2),
            
            # Combate
            'battle_won': self.battle_won,
            'battle_steps': self.battle_steps,
            'pokemon_fainted_player': self.pokemon_fainted_player,
            'pokemon_fainted_opponent': self.pokemon_fainted_opponent,
            
            # Puzzle
            'puzzle_solved': self.puzzle_solved,
            'puzzle_attempts': self.puzzle_attempts,
            'puzzle_steps': self.puzzle_steps,
            
            # Navegación
            'unique_tiles_explored': self.unique_tiles_explored,
            'stuck_count': self.stuck_count,
            'backtrack_count': self.backtrack_count,
            
            # Items
            'potions_used': self.potions_used,
            'status_heals_used': self.status_heals_used,
            'total_items_used': len(self.items_used),
            
            # Equipo
            'final_avg_hp': round(final_hp_avg, 2),
        }
    
    def save_metrics(self, output_dir=None):
        """
        Guarda las métricas en archivos JSON, CSV y Markdown
        
        Args:
            output_dir: Directorio de salida (opcional)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / f"gym{self.gym_number}_{self.gym_name.lower().replace(' ', '_')}" / "results"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = self.session_id
        base_name = f"{self.agent_name}_gym{self.gym_number}_{timestamp}"
        
        # Datos completos
        full_data = {
            'metadata': self.metadata,
            'summary': self.get_summary_stats(),
            'detailed': {
                'reward_history': self.reward_history,
                'action_distribution': self.action_distribution,
                'items_used': self.items_used,
                'game_states': self.game_states,
                'initial_team_hp': self.initial_team_hp,
                'final_team_hp': self.final_team_hp,
            }
        }
        
        # Guardar JSON completo
        json_path = output_dir / f"{base_name}_full.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Guardado JSON: {json_path}")
        
        # Guardar CSV resumido
        csv_path = output_dir / f"{base_name}_summary.csv"
        summary = self.get_summary_stats()
        summary.update(self.metadata)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            writer.writeheader()
            writer.writerow(summary)
        print(f"✓ Guardado CSV: {csv_path}")
        
        # Guardar Markdown detallado
        md_path = output_dir / f"{base_name}_report.md"
        self._save_markdown_report(md_path, full_data)
        print(f"✓ Guardado Markdown: {md_path}")
        
        return {
            'json': json_path,
            'csv': csv_path,
            'markdown': md_path
        }
    
    def _save_markdown_report(self, path, data):
        """Genera reporte en Markdown"""
        summary = data['summary']
        metadata = data['metadata']
        
        success_icon = "✅" if summary['success'] else "❌"
        battle_icon = "🏆" if summary['battle_won'] else "⚔️"
        puzzle_icon = "✅" if summary['puzzle_solved'] else "❓"
        
        report = f"""# 🎮 Reporte de Gimnasio {metadata['gym_number']}: {metadata['gym_name']}

## 📋 Información General
- **Agente:** {metadata['agent_name']}
- **Fecha:** {metadata['timestamp']}
- **Sesión ID:** {metadata['session_id']}
- **Resultado:** {success_icon} {'ÉXITO' if summary['success'] else 'FALLO'}

## ⏱️ Métricas de Tiempo
- **Duración Total:** {summary['total_duration_seconds']:.2f} segundos ({summary['total_duration_seconds']/60:.2f} minutos)
- **Pasos Totales:** {summary['total_steps']:,}
- **Velocidad:** {summary['steps_per_second']:.2f} pasos/segundo

## 🎯 Métricas de Recompensa
- **Recompensa Total:** {summary['total_reward']:.2f}
- **Recompensa Promedio/Paso:** {summary['avg_reward_per_step']:.4f}
- **Recompensa Máxima:** {summary['max_reward']:.2f}
- **Recompensa Mínima:** {summary['min_reward']:.2f}

## ⚔️ Métricas de Combate
- **Estado:** {battle_icon} {'Victoria' if summary['battle_won'] else 'Derrota/No completado'}
- **Duración del Combate:** {summary['battle_steps']} pasos
- **Pokémon Derrotados (Jugador):** {summary['pokemon_fainted_player']}
- **Pokémon Derrotados (Oponente):** {summary['pokemon_fainted_opponent']}

## 🧩 Métricas de Puzzle
- **Estado:** {puzzle_icon} {'Resuelto' if summary['puzzle_solved'] else 'No resuelto'}
- **Intentos:** {summary['puzzle_attempts']}
- **Pasos para resolver:** {summary['puzzle_steps']}

## 🗺️ Métricas de Navegación
- **Baldosas únicas exploradas:** {summary['unique_tiles_explored']}
- **Veces atascado:** {summary['stuck_count']}
- **Retrocesos:** {summary['backtrack_count']}

## 🎒 Uso de Items
- **Pociones usadas:** {summary['potions_used']}
- **Curas de estado usadas:** {summary['status_heals_used']}
- **Total de items usados:** {summary['total_items_used']}

## 💪 Estado del Equipo
- **HP Promedio Final:** {summary['final_avg_hp']:.2f}%

## 📊 Distribución de Acciones
"""
        
        # Agregar distribución de acciones
        action_dist = data['detailed']['action_distribution']
        total_actions = sum(action_dist.values())
        
        action_names = {
            0: "⬇️ Abajo",
            1: "⬅️ Izquierda", 
            2: "➡️ Derecha",
            3: "⬆️ Arriba",
            4: "🅰️ A",
            5: "🅱️ B",
            6: "▶️ Start",
            7: "⏸️ Pass"
        }
        
        for action_id, count in sorted(action_dist.items()):
            percentage = (count / total_actions) * 100
            action_name = action_names.get(action_id, f"Acción {action_id}")
            bar = "█" * int(percentage / 2)
            report += f"- {action_name}: {count:,} ({percentage:.1f}%) {bar}\n"
        
        if summary['failure_reason']:
            report += f"\n## ⚠️ Razón de Fallo\n{summary['failure_reason']}\n"
        
        report += f"\n---\n*Generado automáticamente por GymMetricsTracker*\n"
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)


def compare_agents(metrics_list, output_path=None):
    """
    Compara métricas de múltiples agentes
    
    Args:
        metrics_list: Lista de diccionarios de métricas
        output_path: Path para guardar comparación (opcional)
    
    Returns:
        Diccionario con análisis comparativo
    """
    if not metrics_list:
        return None
    
    comparison = {
        'agents_compared': len(metrics_list),
        'metrics': {}
    }
    
    # Métricas clave para comparar
    key_metrics = [
        'total_duration_seconds',
        'total_steps',
        'total_reward',
        'avg_reward_per_step',
        'success',
        'battle_won',
        'puzzle_solved',
        'unique_tiles_explored'
    ]
    
    for metric in key_metrics:
        values = [m['summary'][metric] for m in metrics_list if metric in m['summary']]
        
        if values:
            comparison['metrics'][metric] = {
                'values': values,
                'mean': np.mean(values) if isinstance(values[0], (int, float)) else None,
                'std': np.std(values) if isinstance(values[0], (int, float)) else None,
                'min': min(values) if isinstance(values[0], (int, float)) else None,
                'max': max(values) if isinstance(values[0], (int, float)) else None,
            }
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)
    
    return comparison
