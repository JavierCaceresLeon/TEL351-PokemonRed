
"""
Script para crear un estado personalizado de Pokemon Red para el gimnasio de Brock.
IMPORTANTE: Parte desde pewter_gym_manual.state (que ya está en el gimnasio) y solo
modifica el equipo/items/dinero. Esto evita corrupciones porque el estado interno
del emulador (VRAM, tiles, registros) ya es correcto para el gimnasio.
"""
from pyboy import PyBoy
from pathlib import Path

# ============================================================================
# DIRECCIONES DE MEMORIA (Pokemon Red)
# ============================================================================

# Equipo Pokemon
PARTY_SIZE = 0xD163
PARTY_SPECIES_LIST = 0xD164

# Estructura de cada Pokemon (44 bytes)
POKEMON_DATA_START = 0xD16B
POKEMON_STRUCT_SIZE = 0x2C

# Offsets
OFF_SPECIES = 0x00
OFF_CURRENT_HP_HI = 0x01
OFF_CURRENT_HP_LO = 0x02
OFF_LEVEL = 0x21
OFF_STATUS = 0x04
OFF_MOVE1 = 0x08
OFF_MOVE2 = 0x09
OFF_MOVE3 = 0x0A
OFF_MOVE4 = 0x0B
OFF_PP1 = 0x1D
OFF_PP2 = 0x1E
OFF_PP3 = 0x1F
OFF_PP4 = 0x20
OFF_MAX_HP_HI = 0x22
OFF_MAX_HP_LO = 0x23
OFF_ATK_HI = 0x24
OFF_ATK_LO = 0x25
OFF_DEF_HI = 0x26
OFF_DEF_LO = 0x27
OFF_SPD_HI = 0x28
OFF_SPD_LO = 0x29
OFF_SPC_HI = 0x2A
OFF_SPC_LO = 0x2B

# Nombres (11 bytes cada uno)
PARTY_OT_NAMES = 0xD273
PARTY_NICKNAMES = 0xD2B5
NAME_LENGTH = 11

# Mochila
BAG_ITEM_COUNT = 0xD31D
BAG_ITEMS_START = 0xD31E

# Dinero y medallas
MONEY_ADDR = 0xD347
BADGES_ADDR = 0xD356

# Mapa (solo para verificacion)
MAP_ID = 0xD35E

# ============================================================================
# DATOS DEL JUEGO
# ============================================================================

POKEMON_IDS = {
    "charmander": 0xB0,
    "pidgey": 0x24,
    "rattata": 0xA5,
}

MOVE_IDS = {
    "scratch": 10,
    "growl": 45,
    "ember": 52,
    "leer": 43,
    "gust": 16,
    "sand_attack": 28,
    "tackle": 33,
    "tail_whip": 39,
    "quick_attack": 98,
}

ITEM_IDS = {
    "potion": 0x14,
    "antidote": 0x18,
    "poke_ball": 0x04,
}

CHAR_MAP = {
    "A": 0x80, "B": 0x81, "C": 0x82, "D": 0x83, "E": 0x84, "F": 0x85,
    "G": 0x86, "H": 0x87, "I": 0x88, "J": 0x89, "K": 0x8A, "L": 0x8B,
    "M": 0x8C, "N": 0x8D, "O": 0x8E, "P": 0x8F, "Q": 0x90, "R": 0x91,
    "S": 0x92, "T": 0x93, "U": 0x94, "V": 0x95, "W": 0x96, "X": 0x97,
    "Y": 0x98, "Z": 0x99, " ": 0x7F,
}
TERMINATOR = 0x50

TEAM_CONFIG = [
    {
        "species": "charmander",
        "level": 12,
        "hp": 33, "max_hp": 33,
        "atk": 15, "def": 12, "spd": 16, "spc": 14,
        "moves": ["scratch", "growl", "ember", "leer"],
        "pps": [35, 40, 25, 30]
    },
    {
        "species": "pidgey",
        "level": 9,
        "hp": 28, "max_hp": 28,
        "atk": 11, "def": 10, "spd": 12, "spc": 9,
        "moves": ["gust", "sand_attack", None, None],
        "pps": [35, 15, 0, 0]
    },
    {
        "species": "rattata",
        "level": 8,
        "hp": 26, "max_hp": 26,
        "atk": 12, "def": 8, "spd": 14, "spc": 7,
        "moves": ["tackle", "tail_whip", "quick_attack", None],
        "pps": [35, 30, 30, 0]
    }
]

BAG_CONFIG = [
    {"item": "potion", "qty": 5},
    {"item": "antidote", "qty": 2},
    {"item": "poke_ball", "qty": 3},
]

MONEY_AMOUNT = 3000

def encode_name(name):
    encoded = []
    for char in name.upper():
        encoded.append(CHAR_MAP.get(char, 0x7F))
    encoded.append(TERMINATOR)
    return encoded

def set_pokemon(pyboy, index, config):
    print(f"Configurando Pokemon {index+1}: {config['species']}...")
    
    # 1. Especie en la lista
    species_id = POKEMON_IDS[config['species']]
    pyboy.memory[PARTY_SPECIES_LIST + index] = species_id
    
    # 2. Estructura de datos
    base_addr = POKEMON_DATA_START + (index * POKEMON_STRUCT_SIZE)
    
    pyboy.memory[base_addr + OFF_SPECIES] = species_id
    pyboy.memory[base_addr + OFF_CURRENT_HP_HI] = (config['hp'] >> 8) & 0xFF
    pyboy.memory[base_addr + OFF_CURRENT_HP_LO] = config['hp'] & 0xFF
    pyboy.memory[base_addr + OFF_LEVEL] = config['level']
    pyboy.memory[base_addr + OFF_STATUS] = 0
    
    # Moves
    for i, move in enumerate(config['moves']):
        move_id = MOVE_IDS.get(move, 0) if move else 0
        pyboy.memory[base_addr + OFF_MOVE1 + i] = move_id
        
    # PPs
    for i, pp in enumerate(config['pps']):
        pyboy.memory[base_addr + OFF_PP1 + i] = pp
        
    # Stats
    pyboy.memory[base_addr + OFF_MAX_HP_HI] = (config['max_hp'] >> 8) & 0xFF
    pyboy.memory[base_addr + OFF_MAX_HP_LO] = config['max_hp'] & 0xFF
    pyboy.memory[base_addr + OFF_ATK_HI] = (config['atk'] >> 8) & 0xFF
    pyboy.memory[base_addr + OFF_ATK_LO] = config['atk'] & 0xFF
    pyboy.memory[base_addr + OFF_DEF_HI] = (config['def'] >> 8) & 0xFF
    pyboy.memory[base_addr + OFF_DEF_LO] = config['def'] & 0xFF
    pyboy.memory[base_addr + OFF_SPD_HI] = (config['spd'] >> 8) & 0xFF
    pyboy.memory[base_addr + OFF_SPD_LO] = config['spd'] & 0xFF
    pyboy.memory[base_addr + OFF_SPC_HI] = (config['spc'] >> 8) & 0xFF
    pyboy.memory[base_addr + OFF_SPC_LO] = config['spc'] & 0xFF
    
    # 3. Nombres (OT y Nickname)
    name_encoded = encode_name(config['species'])
    
    # OT Name (ASH)
    ot_addr = PARTY_OT_NAMES + (index * NAME_LENGTH)
    ash_encoded = encode_name("ASH")
    for i in range(NAME_LENGTH):
        val = ash_encoded[i] if i < len(ash_encoded) else 0x50
        pyboy.memory[ot_addr + i] = val
        
    # Nickname (Species name)
    nick_addr = PARTY_NICKNAMES + (index * NAME_LENGTH)
    for i in range(NAME_LENGTH):
        val = name_encoded[i] if i < len(name_encoded) else 0x50
        pyboy.memory[nick_addr + i] = val

def set_bag(pyboy):
    print("Configurando mochila...")
    pyboy.memory[BAG_ITEM_COUNT] = len(BAG_CONFIG)
    
    for i, item in enumerate(BAG_CONFIG):
        addr = BAG_ITEMS_START + (i * 2)
        pyboy.memory[addr] = ITEM_IDS[item['item']]
        pyboy.memory[addr + 1] = item['qty']
        
    # Limpiar resto (terminator)
    pyboy.memory[BAG_ITEMS_START + (len(BAG_CONFIG) * 2)] = 0xFF

def set_money(pyboy):
    print(f"Configurando dinero: {MONEY_AMOUNT}")
    # BCD encoding (Binary Coded Decimal)
    # 3000 -> 0x00 0x30 0x00
    m_str = f"{MONEY_AMOUNT:06d}"
    b1 = int(m_str[0:2], 16) # Esto es un truco, en realidad BCD es hex visual
    b2 = int(m_str[2:4], 16)
    b3 = int(m_str[4:6], 16)
    
    # BCD real implementation
    b1 = int(m_str[0:2], 10)
    b1 = int(f"{b1:x}", 16) # Convertir dec a hex visual (e.g. 30 -> 0x30)
    
    # Mejor metodo manual para 3000
    pyboy.memory[MONEY_ADDR] = 0x00
    pyboy.memory[MONEY_ADDR + 1] = 0x30
    pyboy.memory[MONEY_ADDR + 2] = 0x00

def main():
    rom_path = "PokemonRed.gb"
    # USAR EL ESTADO MANUAL QUE ACABAS DE CREAR
    base_state = "pewter_gym_manual.state" 
    output_state = "pewter_gym_configured.state"
    
    if not Path(base_state).exists():
        print(f"ERROR: No existe {base_state}. Ejecuta play_manual.py primero.")
        return

    print(f"Cargando base: {base_state}")
    pyboy = PyBoy(rom_path, window='null', sound=False)
    with open(base_state, "rb") as f:
        pyboy.load_state(f)
        
    # Verificar mapa
    map_id = pyboy.memory[MAP_ID]
    print(f"Mapa actual ID: {map_id}")
    if map_id != 54:
        print("ADVERTENCIA: El estado base no parece estar en el gimnasio de Brock (ID 54).")
        print("Continuando de todas formas...")

    # Configurar
    pyboy.memory[PARTY_SIZE] = len(TEAM_CONFIG)
    pyboy.memory[PARTY_SPECIES_LIST + len(TEAM_CONFIG)] = 0xFF # Terminator
    
    for i, pokemon in enumerate(TEAM_CONFIG):
        set_pokemon(pyboy, i, pokemon)
        
    set_bag(pyboy)
    set_money(pyboy)
    
    # Guardar
    print(f"Guardando estado modificado en: {output_state}")
    with open(output_state, "wb") as f:
        pyboy.save_state(f)
        
    pyboy.stop()
    print("Listo!")

if __name__ == "__main__":
    main()
