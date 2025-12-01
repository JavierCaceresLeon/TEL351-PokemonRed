"""
Verificar qué contiene un archivo .state
"""
import argparse
from pathlib import Path
from pyboy import PyBoy

def verify_state(state_path):
    """Verifica el contenido de un archivo .state"""
    print(f"\n🔍 Verificando: {state_path}")
    print("="*60)
    
    # Crear PyBoy con el ROM
    pyboy = PyBoy(
        'PokemonRed.gb',
        window='headless',
        sound=False,
        cgb=False,
        sound_emulated=False
    )
    
    # Cargar el estado
    with open(state_path, 'rb') as f:
        pyboy.load_state(f)
    
    # Leer información crítica
    battle_type = pyboy.memory[0xD057]  # 0=none, 1=wild, 2=trainer
    map_id = pyboy.memory[0xD35E]
    x_pos = pyboy.memory[0xD362]
    y_pos = pyboy.memory[0xD361]
    
    # HP del jugador (2 bytes, big endian)
    player_hp = (pyboy.memory[0xD16C] << 8) | pyboy.memory[0xD16D]
    player_max_hp = (pyboy.memory[0xD18D] << 8) | pyboy.memory[0xD18E]
    
    # HP del enemigo (2 bytes, big endian)
    enemy_hp = (pyboy.memory[0xCFE6] << 8) | pyboy.memory[0xCFE7]
    enemy_max_hp = (pyboy.memory[0xD8F8] << 8) | pyboy.memory[0xD8F9]
    
    badges = pyboy.memory[0xD356]
    
    print(f"\n📍 Posición:")
    print(f"   Map ID: {map_id}")
    print(f"   X: {x_pos}, Y: {y_pos}")
    print(f"   Badges: {badges}")
    
    print(f"\n⚔️  Estado de Batalla:")
    print(f"   Battle Type: {battle_type}", end="")
    if battle_type == 0:
        print(" (NO HAY BATALLA)")
    elif battle_type == 1:
        print(" (Wild Pokemon)")
    elif battle_type == 2:
        print(" (Trainer Battle)")
    else:
        print(" (Desconocido)")
    
    print(f"\n💚 HP Jugador: {player_hp}/{player_max_hp}")
    print(f"💔 HP Enemigo: {enemy_hp}/{enemy_max_hp}")
    
    # Gimnasios conocidos
    gyms = {
        52: "Pewter Gym (Brock)",
        65: "Cerulean Gym (Misty)",
        92: "Vermilion Gym (Lt. Surge)",
        176: "Celadon Gym (Erika)",
        177: "Fuchsia Gym (Koga)",
        178: "Saffron Gym (Sabrina)",
        180: "Cinnabar Gym (Blaine)",
        181: "Viridian Gym (Giovanni)"
    }
    
    if map_id in gyms:
        print(f"\n🏛️  UBICACIÓN: {gyms[map_id]}")
    
    # Diagnóstico
    print(f"\n📊 Diagnóstico:")
    if battle_type > 0:
        print("   ✅ Estado VÁLIDO - Batalla en progreso")
    else:
        print("   ❌ Estado INVÁLIDO - NO hay batalla activa")
        print("   ⚠️  Este estado hará que los modelos deban navegar/explorar")
    
    pyboy.stop()
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verificar contenido de .state')
    parser.add_argument('state', help='Ruta al archivo .state')
    args = parser.parse_args()
    
    verify_state(args.state)
