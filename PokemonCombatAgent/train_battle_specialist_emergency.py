"""
Emergency Battle Specialist Training - SOLO batallas de gimnasio
Objetivo: Superar al PPO baseline en WIN RATE y EFICIENCIA en batallas
Tiempo estimado: 2-3 horas de entrenamiento
"""

import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import random

from combat_focused_env import CombatFocusedEnv, make_combat_env_config


def create_battle_specialist_config(battle_state, session_path):
    """Config optimizado para batallas puras"""
    return {
        'headless': True,
        'save_final_state': False,
        'early_stop': False,
        'action_freq': 24,
        'init_state': battle_state,
        'max_steps': 2000,  # Batallas son cortas
        'print_rewards': False,
        'save_video': False,
        'fast_video': False,
        'session_path': session_path,
        'gb_path': 'PokemonRed.gb',
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'extra_buttons': False
    }


class BattleSpecialistEnv(CombatFocusedEnv):
    """
    Especialista PURO en batallas
    Recompensas EXTREMAS para ganar rápido
    """
    
    def _calculate_combat_reward(self):
        """Recompensas AGRESIVAS para batallas"""
        battle_type = self.read_m(0xD057)
        reward = 0
        
        if battle_type > 0:  # En batalla
            # Leer HP actual
            current_player_hp = self.read_hp(0xD16C)
            current_enemy_hp = self.read_hp(0xCFE6)
            
            # Primera vez en batalla - guardar HP inicial
            if not hasattr(self, '_battle_active'):
                self._battle_active = True
                self._initial_player_hp = current_player_hp
                self._initial_enemy_hp = current_enemy_hp
                self._last_enemy_hp = current_enemy_hp
                return 10.0  # Bonus por entrar a batalla
            
            # Daño causado este step
            damage_dealt = self._last_enemy_hp - current_enemy_hp
            if damage_dealt > 0:
                reward += damage_dealt * 2.0  # DOBLE recompensa por daño
            
            # Daño recibido
            damage_taken = self._initial_player_hp - current_player_hp
            if damage_taken > 0:
                reward -= damage_taken * 0.5
            
            # Bonus por estar atacando (HP enemigo bajo)
            enemy_hp_ratio = current_enemy_hp / max(self._initial_enemy_hp, 1)
            if enemy_hp_ratio < 0.5:
                reward += 5.0  # Bonus por dominar
            
            self._last_enemy_hp = current_enemy_hp
            
        else:  # Fuera de batalla
            if hasattr(self, '_battle_active'):
                # Batalla terminó
                if hasattr(self, '_last_enemy_hp') and self._last_enemy_hp == 0:
                    # VICTORIA!
                    reward += 500.0  # MEGA BONUS
                    player_hp_remaining = self.read_hp(0xD16C)
                    # Bonus por eficiencia (ganar con mucha vida)
                    hp_ratio = player_hp_remaining / max(self._initial_player_hp, 1)
                    reward += hp_ratio * 100.0
                else:
                    # Perdió o huyó
                    reward -= 200.0
                
                # Reset
                del self._battle_active
                del self._initial_player_hp
                del self._initial_enemy_hp
                del self._last_enemy_hp
        
        return reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--battle-state', type=str, default='battle_states/pewter_battle.state',
                        help='Battle state to specialize in')
    parser.add_argument('--timesteps', type=int, default=200_000,
                        help='Training timesteps (200K ~2 horas)')
    parser.add_argument('--session-name', type=str, default='battle_specialist')
    
    args = parser.parse_args()
    
    # Setup
    session_path = Path('sessions') / args.session_name
    session_path.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("🥊 EMERGENCY BATTLE SPECIALIST TRAINING")
    print("="*60)
    print(f"Battle: {args.battle_state}")
    print(f"Target: {args.timesteps:,} timesteps (~2 hours)")
    print(f"Device: {device}")
    print(f"Objective: DOMINATE gym battles!")
    print("="*60)
    print()
    
    # Create specialized environment
    config = create_battle_specialist_config(args.battle_state, session_path)
    env = BattleSpecialistEnv(config)
    
    # Aggressive PPO settings for fast learning
    model = PPO(
        'MultiInputPolicy',
        env,
        verbose=1,
        n_steps=1024,  # Más pequeño para batallas rápidas
        batch_size=64,
        n_epochs=10,  # MÁS epochs para aprender rápido
        gamma=0.95,  # Menos foco en futuro lejano
        learning_rate=0.0005,  # MÁS rápido
        ent_coef=0.02,  # MÁS exploración
        device=device,
        tensorboard_log=str(session_path)
    )
    
    # Checkpoint cada 25K
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=str(session_path / 'checkpoints'),
        name_prefix='battle_specialist',
        save_replay_buffer=False
    )
    
    print("🚀 Starting AGGRESSIVE training...\n")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        final_path = session_path / f'{args.session_name}_final.zip'
        model.save(str(final_path))
        
        print(f"\n{'='*60}")
        print(f"✅ Battle Specialist trained!")
        print(f"📁 Model: {final_path}")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        interrupt_path = session_path / f'{args.session_name}_interrupted.zip'
        model.save(str(interrupt_path))
        print(f"\n⚠️ Saved to: {interrupt_path}")
    
    finally:
        env.close()


if __name__ == '__main__':
    main()
