"""
Script para inspeccionar el contenido de un archivo .state de PyBoy
"""

from pyboy import PyBoy
import sys

def inspect_state(state_file, gb_file='PokemonRed.gb'):
    """Carga el state y muestra información sobre el estado del juego."""
    
    print(f"Cargando {gb_file}...")
    pyboy = PyBoy(gb_file, window="null")
    
    print(f"Cargando estado: {state_file}...")
    with open(state_file, 'rb') as f:
        pyboy.load_state(f)
    
    memory = pyboy.memory
    
    print("\n" + "="*70)
    print("ESTADO DEL JUEGO")
    print("="*70)
    
    # Información de batalla
    in_battle = memory[0xD057]
    battle_type = memory[0xD05A]
    
    print(f"\nEn batalla: {in_battle != 0}")
    print(f"Tipo de batalla: {battle_type}")
    
    # HP del jugador (Pokemon activo en party slot 1)
    party_count = memory[0xD163]
    print(f"Pokemon en party: {party_count}")
    
    # Pokemon del jugador
    player_poke_hp = (memory[0xD016] << 8) | memory[0xD017]
    player_poke_max_hp = (memory[0xD018] << 8) | memory[0xD019]
    player_poke_level = memory[0xD01C]
    player_poke_species = memory[0xD014]
    
    print(f"\nPokemon del jugador:")
    print(f"  Especie ID: {player_poke_species}")
    print(f"  Nivel: {player_poke_level}")
    print(f"  HP: {player_poke_hp}/{player_poke_max_hp}")
    
    # Pokemon enemigo
    enemy_hp = (memory[0xCFE7] << 8) | memory[0xCFE8]
    enemy_max_hp = (memory[0xCFE9] << 8) | memory[0xCFEA]
    enemy_level = memory[0xCFF3]
    enemy_species = memory[0xCFE5]
    
    print(f"\nPokemon enemigo:")
    print(f"  Especie ID: {enemy_species}")
    print(f"  Nivel: {enemy_level}")
    print(f"  HP: {enemy_hp}/{enemy_max_hp}")
    
    # Estado de texto/menú
    text_box_active = memory[0xCC57]
    menu_selection = memory[0xCC24]
    battle_menu_selection = memory[0xCC2F]
    
    print(f"\nEstado de interfaz:")
    print(f"  Texto activo: {text_box_active}")
    print(f"  Selección menú: {menu_selection}")
    print(f"  Selección menú batalla: {battle_menu_selection}")
    
    # Posición en el mundo
    map_n = memory[0xD35E]
    x_pos = memory[0xD362]
    y_pos = memory[0xD361]
    
    print(f"\nPosición:")
    print(f"  Mapa: {map_n}")
    print(f"  X: {x_pos}, Y: {y_pos}")
    
    # Flags de juego importantes
    obtained_pokedex = memory[0xD356]
    badges = memory[0xD356]
    
    print(f"\nProgreso:")
    print(f"  Pokedex obtenido: {obtained_pokedex & 0x01}")
    print(f"  Badges: {bin(badges)}")
    
    # Captura de pantalla
    print(f"\nTomando captura de pantalla...")
    screen_image = pyboy.screen.image
    screen_image.save('state_screenshot.png')
    print(f"✓ Guardada como state_screenshot.png")
    
    pyboy.stop()
    print("\n" + "="*70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=str, 
                       default='generated_battle_states/clean_pewter_gym.state')
    parser.add_argument('--gb', type=str, default='PokemonRed.gb')
    
    args = parser.parse_args()
    
    inspect_state(args.state, args.gb)
