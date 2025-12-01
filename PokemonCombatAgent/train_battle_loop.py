"""
Continuar entrenamiento del Combat Agent en LOOP de batalla
- Usa el estado de batalla válido (clean_pewter_gym.state)
- Repite la misma batalla una y otra vez
- Recompensas optimizadas para eficiencia en combate:
  * Ganar batalla: +1000
  * Daño causado: +3.0 por HP
  * Daño recibido: -2.0 por HP
  * Huir con mucha vida: -500 (penalización)
  * Victoria sin daño (perfect): +300 bonus
  * Derrota: -300
"""
import argparse
from pathlib import Path
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym

# Import environment
from red_gym_env_v2 import RedGymEnv
from battle_only_actions import BattleOnlyActions

class BattleLoopEnv(gym.Wrapper):
    """
    Wrapper que optimiza recompensas para entrenamiento en loop de batalla.
    Penaliza huir con mucha vida y premia eficiencia.
    """
    def __init__(self, env):
        super().__init__(env)
        self.last_player_hp = 0
        self.last_enemy_hp = 0
        self.initial_player_hp = 0
        self.initial_enemy_hp = 0
        self.battle_start = True
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Leer HP inicial
        self.initial_player_hp = self._read_hp(0xD16C)
        self.initial_enemy_hp = self._read_hp(0xCFE6)
        self.last_player_hp = self.initial_player_hp
        self.last_enemy_hp = self.initial_enemy_hp
        self.battle_start = True
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        
        print(f"\n🔄 Nueva batalla | Player HP: {self.initial_player_hp} | Enemy HP: {self.initial_enemy_hp}")
        
        return obs, info
    
    def _read_hp(self, address):
        """Lee HP de 2 bytes (big endian)"""
        try:
            high = self.env.pyboy.memory[address]
            low = self.env.pyboy.memory[address + 1]
            return (high << 8) | low
        except:
            return 0
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Leer HP actual
        current_player_hp = self._read_hp(0xD16C)
        current_enemy_hp = self._read_hp(0xCFE6)
        battle_type = self.env.pyboy.memory[0xD057]
        
        # Calcular cambios de HP
        damage_dealt = max(0, self.last_enemy_hp - current_enemy_hp)
        damage_taken = max(0, self.last_player_hp - current_player_hp)
        
        # Acumular totales
        self.total_damage_dealt += damage_dealt
        self.total_damage_taken += damage_taken
        
        # RECOMPENSAS PERSONALIZADAS
        combat_reward = 0
        
        # 1. Daño causado (muy positivo)
        if damage_dealt > 0:
            combat_reward += damage_dealt * 3.0
            print(f"  💥 Causó {damage_dealt} daño → +{damage_dealt * 3.0:.1f}")
        
        # 2. Daño recibido (penalización moderada)
        if damage_taken > 0:
            combat_reward -= damage_taken * 2.0
            print(f"  💔 Recibió {damage_taken} daño → -{damage_taken * 2.0:.1f}")
        
        # 3. Detectar fin de batalla
        if battle_type == 0 and self.last_battle_type > 0:
            # Batalla terminó
            player_hp_percent = current_player_hp / max(1, self.initial_player_hp)
            
            resultado = ""
            if current_enemy_hp == 0:
                # VICTORIA
                combat_reward += 1000
                
                # Bonus por victoria perfecta (sin daño)
                if self.total_damage_taken == 0:
                    combat_reward += 300
                    resultado = "🏆 VICTORIA PERFECTA"
                    print(f"  {resultado}! → +1300 | Total reward: {combat_reward:.1f}")
                else:
                    resultado = "✅ VICTORIA"
                    print(f"  {resultado}! → +1000 | Total reward: {combat_reward:.1f}")
                    
            elif player_hp_percent > 0.5:
                # HUYÓ con más del 50% HP (PENALIZACIÓN FUERTE)
                combat_reward -= 500
                resultado = "🏃 HUYÓ"
                print(f"  {resultado} con {player_hp_percent*100:.0f}% HP → -500 | Total reward: {combat_reward:.1f}")
                
            elif current_player_hp == 0:
                # DERROTA
                combat_reward -= 300
                resultado = "☠️  DERROTA"
                print(f"  {resultado} → -300 | Total reward: {combat_reward:.1f}")
            
            # Mostrar estadísticas de la batalla
            print(f"  📊 Stats | Daño dado: {self.total_damage_dealt} | Daño recibido: {self.total_damage_taken}")
            
            # Resetear para próxima batalla (ESTO ES LO QUE QUEREMOS)
            # PPO aprende jugando MUCHAS batallas, no una sola batalla larga
            terminated = True
        
        # Actualizar estado previo
        self.last_player_hp = current_player_hp
        self.last_enemy_hp = current_enemy_hp
        self.last_battle_type = battle_type
        
        # Reemplazar reward original con combat_reward
        final_reward = combat_reward
        
        return obs, final_reward, terminated, truncated, info

def create_battle_env(battle_state_path):
    """Crea el ambiente de entrenamiento en loop de batalla"""
    config = {
        'headless': True,
        'save_final_state': False,
        'early_stop': False,
        'action_freq': 24,
        'init_state': battle_state_path,
        'max_steps': 500,  # Límite por batalla (evita loops infinitos)
        'print_rewards': False,
        'save_video': False,
        'fast_video': False,
        'session_path': Path('battle_loop_session'),
        'gb_path': 'PokemonRed.gb',
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'extra_buttons': False
    }
    
    env = RedGymEnv(config)
    env = BattleOnlyActions(env)  # Reducir acciones a solo las válidas en batalla
    env = BattleLoopEnv(env)
    
    return env

def main():
    parser = argparse.ArgumentParser(description='Entrenar Combat Agent en loop de batalla')
    parser.add_argument('--model', required=True, help='Modelo a continuar entrenando (.zip)')
    parser.add_argument('--battle-state', default='generated_battle_states/clean_pewter_gym.state',
                        help='Estado de batalla para entrenar')
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Timesteps de entrenamiento adicional')
    parser.add_argument('--learning-rate', type=float, default=0.0003,
                        help='Learning rate')
    args = parser.parse_args()
    
    print("="*60)
    print("ENTRENAMIENTO EN LOOP DE BATALLA")
    print("="*60)
    print(f"Modelo base: {args.model}")
    print(f"Estado batalla: {args.battle_state}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Learning rate: {args.learning_rate}")
    print("\nSistema de recompensas:")
    print("  • Daño causado: +3.0 por HP")
    print("  • Daño recibido: -2.0 por HP")
    print("  • Victoria: +1000")
    print("  • Victoria perfecta (sin daño): +1300")
    print("  • Huir con >50% HP: -500")
    print("  • Derrota: -300")
    print("="*60)
    
    # Verificar que el modelo existe
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n❌ ERROR: Modelo no encontrado: {args.model}")
        return
    
    # Verificar que el estado existe
    state_path = Path(args.battle_state)
    if not state_path.exists():
        print(f"\n❌ ERROR: Estado no encontrado: {args.battle_state}")
        return
    
    # Crear ambiente
    print("\n📦 Creando ambiente de batalla...")
    env = create_battle_env(str(state_path))
    
    # Nombre de sesión (usar siempre, incluso si creamos un modelo nuevo)
    session_name = model_path.stem + "_battle_loop"

    # Intentar cargar modelo pre-entrenado
    try:
        print(f"📥 Intentando cargar modelo desde {args.model}...")
        model = PPO.load(args.model, env=env)
        print("✅ Modelo cargado exitosamente (espacios de acción coinciden)")
        
        # Actualizar learning rate si es diferente
        model.learning_rate = args.learning_rate
        
    except ValueError as e:
        if "Action spaces do not match" in str(e):
            print(f"\n⚠️  Error: {e}")
            print("\n🆕 Creando modelo NUEVO desde cero (espacios de acción incompatibles)")
            print("   Esto es normal si cambiaste el espacio de acciones (BattleOnlyActions)")
            
            # Crear modelo nuevo
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                n_steps=2048,
                batch_size=512,
                n_epochs=4,
                learning_rate=args.learning_rate,
                tensorboard_log=f"sessions/{session_name}"
            )
            print("✅ Modelo nuevo creado")
        else:
            raise
    
    # Configurar checkpoints
    checkpoint_dir = Path('sessions') / session_name / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(checkpoint_dir),
        name_prefix=session_name
    )
    
    # Entrenar
    print(f"\n🚀 Iniciando entrenamiento por {args.timesteps:,} timesteps...")
    print("Ctrl+C para detener y guardar\n")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False  # Continuar contador de timesteps
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Entrenamiento interrumpido por usuario")
    
    # Guardar modelo final
    output_path = Path('sessions') / session_name / f"{session_name}.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Guardando modelo entrenado en: {output_path}")
    model.save(output_path)
    
    print("\n✅ Entrenamiento completado!")
    print(f"Modelo guardado: {output_path}")
    print(f"Checkpoints en: {checkpoint_dir}")
    
    env.close()

if __name__ == '__main__':
    main()
