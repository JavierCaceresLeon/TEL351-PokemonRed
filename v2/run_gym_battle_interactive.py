"""
Script interactivo para probar el agente entrenado en batallas de gimnasio.
Permite cargar estados de gimnasio y modificar el equipo Pokémon e items.
"""
import os
from os.path import exists
from pathlib import Path
import uuid
import sys

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker

# ============================================================================
# CONFIGURACIÓN - MODIFICA AQUÍ
# ============================================================================

# Estado a cargar (opciones: pewter_battle, cerulean_battle, celadon_battle, etc.)
GYM_STATE = "pewter_battle"

# Checkpoint del modelo entrenado
MODEL_PATH = r"C:\Users\Cris\Documents\GitHub\TEL351-PokemonRed\PokemonCombatAgent\combat_agent_final_continued.zip"

# ============================================================================
# DIRECCIONES DE MEMORIA PARA MODIFICAR EQUIPO
# ============================================================================

# Estructura del equipo
PARTY_SIZE_ADDRESS = 0xD163
PARTY_SPECIES = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
PARTY_LEVELS = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
PARTY_HP_CURRENT = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
PARTY_HP_MAX = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]

# Mochila
BAG_ITEM_COUNT = 0xD31D
BAG_ITEMS_START = 0xD31E

# IDs de Pokémon útiles
POKEMON_IDS = {
    'bulbasaur': 0x99, 'ivysaur': 0x09, 'venusaur': 0x9A,
    'charmander': 0xB0, 'charmeleon': 0xB2, 'charizard': 0xB4,
    'squirtle': 0xB1, 'wartortle': 0xB3, 'blastoise': 0x1C,
    'pikachu': 0x54, 'raichu': 0x55,
    'pidgey': 0x24, 'pidgeotto': 0x96, 'pidgeot': 0x97,
    'rattata': 0xA5, 'raticate': 0xA6,
    'spearow': 0x05, 'fearow': 0x23,
    'nidoran_m': 0x03, 'nidorino': 0xA7, 'nidoking': 0x07,
    'nidoran_f': 0x0F, 'nidorina': 0xA8, 'nidoqueen': 0x10,
    'clefairy': 0x04, 'clefable': 0x8E,
    'jigglypuff': 0x64, 'wigglytuff': 0x65,
    'zubat': 0x6B, 'golbat': 0x82,
    'oddish': 0xB9, 'gloom': 0xBA, 'vileplume': 0xBB,
    'paras': 0x6D, 'parasect': 0x2E,
    'diglett': 0x3B, 'dugtrio': 0x76,
    'meowth': 0x4D, 'persian': 0x90,
    'psyduck': 0x2F, 'golduck': 0x80,
    'mankey': 0x39, 'primeape': 0x75,
    'growlithe': 0x21, 'arcanine': 0x14,
    'poliwag': 0x47, 'poliwhirl': 0x6E, 'poliwrath': 0x6F,
    'abra': 0x94, 'kadabra': 0x26, 'alakazam': 0x95,
    'machop': 0x6A, 'machoke': 0x29, 'machamp': 0x7E,
    'bellsprout': 0xBC, 'weepinbell': 0xBD, 'victreebel': 0xBE,
    'tentacool': 0x18, 'tentacruel': 0x9B,
    'geodude': 0xA9, 'graveler': 0x27, 'golem': 0x31,
    'ponyta': 0xA3, 'rapidash': 0xA4,
    'magnemite': 0xAD, 'magneton': 0x36,
    'farfetchd': 0x40, 'doduo': 0x46, 'dodrio': 0x74,
    'seel': 0x3A, 'dewgong': 0x78,
    'grimer': 0x0D, 'muk': 0x88,
    'shellder': 0x17, 'cloyster': 0x8B,
    'gastly': 0x19, 'haunter': 0x93, 'gengar': 0x0E,
    'onix': 0x22,
    'drowzee': 0x30, 'hypno': 0x81,
    'krabby': 0x4E, 'kingler': 0x8A,
    'voltorb': 0x06, 'electrode': 0x8D,
    'exeggcute': 0x0C, 'exeggutor': 0x0A,
    'cubone': 0x11, 'marowak': 0x91,
    'hitmonlee': 0x2B, 'hitmonchan': 0x2C,
    'lickitung': 0x0B,
    'koffing': 0x37, 'weezing': 0x8F,
    'rhyhorn': 0x12, 'rhydon': 0x01,
    'chansey': 0x28,
    'tangela': 0x1E,
    'kangaskhan': 0x02,
    'horsea': 0x5C, 'seadra': 0x5D,
    'goldeen': 0x9D, 'seaking': 0x9E,
    'staryu': 0x1B, 'starmie': 0x98,
    'mr_mime': 0x2A,
    'scyther': 0x1A,
    'jynx': 0x48,
    'electabuzz': 0x35,
    'magmar': 0x33,
    'pinsir': 0x1D,
    'tauros': 0x3C,
    'magikarp': 0x85, 'gyarados': 0x16,
    'lapras': 0x13,
    'ditto': 0x4C,
    'eevee': 0x66, 'vaporeon': 0x69, 'jolteon': 0x68, 'flareon': 0x67,
    'porygon': 0xAA,
    'omanyte': 0x62, 'omastar': 0x63,
    'kabuto': 0x5A, 'kabutops': 0x5B,
    'aerodactyl': 0xAB,
    'snorlax': 0x84,
    'articuno': 0x4A, 'zapdos': 0x4B, 'moltres': 0x49,
    'dratini': 0x58, 'dragonair': 0x59, 'dragonite': 0x42,
    'mewtwo': 0x83, 'mew': 0x15,
}

# IDs de Items útiles
ITEM_IDS = {
    'potion': 0x14,
    'super_potion': 0x15,
    'hyper_potion': 0x16,
    'max_potion': 0x17,
    'full_restore': 0x1D,
    'revive': 0x1F,
    'max_revive': 0x20,
    'x_attack': 0x31,
    'x_defend': 0x32,
    'x_speed': 0x33,
    'x_special': 0x34,
    'poke_ball': 0x04,
    'great_ball': 0x05,
    'ultra_ball': 0x06,
    'master_ball': 0x01,
}

# ============================================================================
# FUNCIONES PARA MODIFICAR MEMORIA
# ============================================================================

def write_memory(pyboy, address, value):
    """Escribe un byte en la memoria."""
    pyboy.memory[address] = value

def write_memory_word(pyboy, address, value):
    """Escribe un word (2 bytes, big endian) en la memoria."""
    pyboy.memory[address] = (value >> 8) & 0xFF
    pyboy.memory[address + 1] = value & 0xFF

def read_memory(pyboy, address):
    """Lee un byte de la memoria."""
    return pyboy.memory[address]

def set_pokemon(pyboy, slot, species_name, level, hp=None):
    """
    Configura un Pokémon en el equipo.
    
    Args:
        pyboy: Instancia de PyBoy
        slot: Posición en el equipo (0-5)
        species_name: Nombre del Pokémon (ver POKEMON_IDS)
        level: Nivel (1-100)
        hp: HP actual (si None, usa HP máximo)
    """
    if slot < 0 or slot > 5:
        print(f"Error: slot debe estar entre 0 y 5")
        return
    
    species_id = POKEMON_IDS.get(species_name.lower())
    if species_id is None:
        print(f"Error: Pokémon '{species_name}' no encontrado")
        print(f"Disponibles: {list(POKEMON_IDS.keys())}")
        return
    
    # Asegurar que el tamaño del equipo incluya este slot
    current_size = read_memory(pyboy, PARTY_SIZE_ADDRESS)
    if slot >= current_size:
        write_memory(pyboy, PARTY_SIZE_ADDRESS, slot + 1)
    
    # Escribir especie
    write_memory(pyboy, PARTY_SPECIES[slot], species_id)
    
    # Escribir nivel
    write_memory(pyboy, PARTY_LEVELS[slot], level)
    
    # Calcular HP aproximado basado en nivel (simplificado)
    base_hp = 50 + level * 2  # Aproximación
    if hp is None:
        hp = base_hp
    
    # Escribir HP actual y máximo
    write_memory_word(pyboy, PARTY_HP_CURRENT[slot], hp)
    write_memory_word(pyboy, PARTY_HP_MAX[slot], base_hp)
    
    print(f"✓ Slot {slot}: {species_name.capitalize()} Lv.{level} HP:{hp}/{base_hp}")

def set_items(pyboy, items):
    """
    Configura los items de la mochila.
    
    Args:
        pyboy: Instancia de PyBoy
        items: Lista de tuplas (item_name, cantidad)
    """
    # Escribir número de items
    write_memory(pyboy, BAG_ITEM_COUNT, len(items))
    
    # Escribir cada item
    for i, (item_name, qty) in enumerate(items):
        item_id = ITEM_IDS.get(item_name.lower())
        if item_id is None:
            print(f"Warning: Item '{item_name}' no encontrado, saltando...")
            continue
        
        addr = BAG_ITEMS_START + (i * 2)
        write_memory(pyboy, addr, item_id)
        write_memory(pyboy, addr + 1, qty)
        print(f"✓ Item {i}: {item_name} x{qty}")
    
    # Terminar la lista de items
    write_memory(pyboy, BAG_ITEMS_START + (len(items) * 2), 0xFF)

def show_current_team(pyboy):
    """Muestra el equipo actual."""
    size = read_memory(pyboy, PARTY_SIZE_ADDRESS)
    print(f"\n=== Equipo Actual ({size} Pokémon) ===")
    
    # Diccionario inverso para nombres
    id_to_name = {v: k for k, v in POKEMON_IDS.items()}
    
    for i in range(min(size, 6)):
        species = read_memory(pyboy, PARTY_SPECIES[i])
        level = read_memory(pyboy, PARTY_LEVELS[i])
        name = id_to_name.get(species, f"ID:{hex(species)}")
        print(f"  Slot {i}: {name.capitalize()} Lv.{level}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Construir path al estado del gimnasio
    state_path = parent_dir / "gym_scenarios" / "state_files" / f"{GYM_STATE}.state"
    
    if not state_path.exists():
        print(f"ERROR: Estado no encontrado: {state_path}")
        print("\nEstados disponibles:")
        for f in (parent_dir / "gym_scenarios" / "state_files").glob("*.state"):
            print(f"  - {f.stem}")
        sys.exit(1)
    
    print(f"=== Cargando estado: {GYM_STATE} ===")
    print(f"Estado: {state_path}")
    print(f"Modelo: {MODEL_PATH}")
    
    sess_path = Path(f'session_gym_{str(uuid.uuid4())[:8]}')
    ep_length = 2**20  # Más corto para testing
    
    env_config = {
        'headless': False,
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': str(state_path),
        'max_steps': ep_length,
        'print_rewards': True,
        'save_video': False,
        'fast_video': True,
        'session_path': sess_path,
        'gb_path': str(parent_dir / 'PokemonRed.gb'),
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'extra_buttons': False
    }
    
    # Crear entorno
    env = RedGymEnv(env_config)
    
    # Verificar entorno
    print("\nVerificando entorno...")
    env_checker.check_env(env)
    print("✓ Entorno válido")
    
    # Mostrar equipo actual
    show_current_team(env.pyboy)
    
    # ========================================================================
    # MODIFICA EL EQUIPO AQUÍ (descomenta y ajusta según necesites)
    # ========================================================================
    
    # Ejemplo: Configurar un equipo personalizado
    # set_pokemon(env.pyboy, 0, 'charizard', 36)
    # set_pokemon(env.pyboy, 1, 'pikachu', 30)
    # set_pokemon(env.pyboy, 2, 'alakazam', 35)
    
    # Ejemplo: Configurar items
    # set_items(env.pyboy, [
    #     ('hyper_potion', 10),
    #     ('full_restore', 5),
    #     ('revive', 3),
    #     ('x_attack', 5),
    # ])
    
    # ========================================================================
    
    # Cargar modelo
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Modelo no encontrado: {MODEL_PATH}")
        sys.exit(1)
    
    print(f"\nCargando modelo...")
    model = PPO.load(MODEL_PATH, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
    print("✓ Modelo cargado")
    
    # Crear archivo de control si no existe
    agent_file = current_dir / "agent_enabled.txt"
    if not agent_file.exists():
        agent_file.write_text("yes")
    
    print("\n" + "="*50)
    print("CONTROLES:")
    print("  - Edita 'agent_enabled.txt' con 'yes' o 'no'")
    print("  - 'yes' = el agente juega automáticamente")
    print("  - 'no' = puedes jugar manualmente")
    print("="*50 + "\n")
    
    # Loop principal
    obs, info = env.reset()
    
    while True:
        try:
            with open(agent_file, "r") as f:
                agent_enabled = f.read().strip().lower().startswith("yes")
        except:
            agent_enabled = False
        
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, terminated, truncated, info = env.step(action)
        else:
            env.pyboy.tick(1, True)
            obs = env._get_obs()
            truncated = env.step_count >= env.max_steps - 1
        
        env.render()
        
        if truncated:
            break
    
    env.close()
    print("\n✓ Sesión terminada")
