
"""
Script para crear un estado personalizado de Pokemon Red para el gimnasio de Brock.
Basado en un estado valido de Pewter Gym para evitar corrupcion grafica/mapa.
"""
from pyboy import PyBoy
from pathlib import Path
import os

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
OFF_TYPE1 = 0x05
OFF_TYPE2 = 0x06
OFF_CATCH_RATE = 0x07
OFF_MOVE1 = 0x08
OFF_MOVE2 = 0x09
OFF_MOVE3 = 0x0A
OFF_MOVE4 = 0x0B
OFF_TRAINER_ID_HI = 0x0C
OFF_TRAINER_ID_LO = 0x0D
OFF_EXP_HI = 0x0E
OFF_EXP_MID = 0x0F
OFF_EXP_LO = 0x10
OFF_HP_EV_HI = 0x11
OFF_HP_EV_LO = 0x12
OFF_ATK_EV_HI = 0x13
OFF_ATK_EV_LO = 0x14
OFF_DEF_EV_HI = 0x15
OFF_DEF_EV_LO = 0x16
OFF_SPD_EV_HI = 0x17
OFF_SPD_EV_LO = 0x18
OFF_SPC_EV_HI = 0x19
OFF_SPC_EV_LO = 0x1A
OFF_ATK_DEF_IV = 0x1B
OFF_SPD_SPC_IV = 0x1C
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

# Mochila
BAG_ITEM_COUNT = 0xD31D
BAG_ITEMS_START = 0xD31E

# Dinero
MONEY_ADDR = 0xD347

# Medallas
BADGES_ADDR = 0xD356

# Posicion
MAP_ID = 0xD35E
Y_POS = 0xD361
X_POS = 0xD362

# ============================================================================
# DATOS
# ============================================================================

POKEMON_IDS = {
    'charmander': 0xB0,
    'pidgey': 0x24,
    'rattata': 0xA5,
}

MOVE_IDS = {
    'scratch': 10,
    'growl': 45,
    'ember': 52,
    'leer': 43,
    'gust': 16,
    'sand_attack': 28,
    'tackle': 33,
    'tail_whip': 39,
    'quick_attack': 98,
}

ITEM_IDS = {
    'potion': 0x14,
    'antidote': 0x18,
    'poke_ball': 0x04,
}

# Configuracion exacta solicitada
TEAM_CONFIG = [
    {
        'species': 'charmander',
        'level': 12,
        'current_hp': 33,
        'max_hp': 33,
        'attack': 15,
        'defense': 12,
        'speed': 16,
        'special': 14,
        'moves': [
            ('scratch', 35),
            ('growl', 40),
            ('ember', 25),
            ('leer', 30),
        ]
    },
    {
        'species': 'pidgey',
        'level': 9,
        'current_hp': 28,
        'max_hp': 28,
        'attack': 11,
        'defense': 10,
        'speed': 12,
        'special': 9,
        'moves': [
            ('gust', 35),
            ('sand_attack', 15),
        ]
    },
    {
        'species': 'rattata',
        'level': 8,
        'current_hp': 26,
        'max_hp': 26,
        'attack': 12,
        'defense': 8,
        'speed': 14,
        'special': 7,
        'moves': [
            ('tackle', 35),
            ('tail_whip', 30),
            ('quick_attack', 30),
        ]
    },
]

BAG_CONFIG = [
    ('potion', 5),
    ('antidote', 2),
    ('poke_ball', 3),
]

MONEY = 3000
BADGES = 0

# ============================================================================
# FUNCIONES
# ============================================================================

def write_byte(pyboy, addr, value):
    pyboy.memory[addr] = value & 0xFF

def write_word_be(pyboy, addr, value):
    pyboy.memory[addr] = (value >> 8) & 0xFF
    pyboy.memory[addr + 1] = value & 0xFF

def decimal_to_bcd(value):
    value = max(0, min(999999, value))
    b1 = ((value // 100000) << 4) | ((value // 10000) % 10)
    b2 = (((value // 1000) % 10) << 4) | ((value // 100) % 10)
    b3 = (((value // 10) % 10) << 4) | (value % 10)
    return b1, b2, b3

def set_money(pyboy, amount):
    b1, b2, b3 = decimal_to_bcd(amount)
    write_byte(pyboy, MONEY_ADDR, b1)
    write_byte(pyboy, MONEY_ADDR + 1, b2)
    write_byte(pyboy, MONEY_ADDR + 2, b3)

def set_pokemon(pyboy, slot, config):
    species_id = POKEMON_IDS[config['species']]
    base_addr = POKEMON_DATA_START + (slot * POKEMON_STRUCT_SIZE)
    
    write_byte(pyboy, PARTY_SPECIES_LIST + slot, species_id)
    write_byte(pyboy, base_addr + OFF_SPECIES, species_id)
    write_word_be(pyboy, base_addr + OFF_CURRENT_HP_HI, config['current_hp'])
    write_byte(pyboy, base_addr + OFF_LEVEL, config['level'])
    
    for i in range(4):
        if i < len(config['moves']):
            move_name, pp = config['moves'][i]
            move_id = MOVE_IDS.get(move_name, 0)
            write_byte(pyboy, base_addr + OFF_MOVE1 + i, move_id)
            write_byte(pyboy, base_addr + OFF_PP1 + i, pp)
        else:
            write_byte(pyboy, base_addr + OFF_MOVE1 + i, 0)
            write_byte(pyboy, base_addr + OFF_PP1 + i, 0)
    
    write_word_be(pyboy, base_addr + OFF_MAX_HP_HI, config['max_hp'])
    write_word_be(pyboy, base_addr + OFF_ATK_HI, config['attack'])
    write_word_be(pyboy, base_addr + OFF_DEF_HI, config['defense'])
    write_word_be(pyboy, base_addr + OFF_SPD_HI, config['speed'])
    write_word_be(pyboy, base_addr + OFF_SPC_HI, config['special'])
    write_byte(pyboy, base_addr + OFF_STATUS, 0)
    
    print(f"  Slot {slot + 1}: {config['species'].capitalize()} Lv.{config['level']}")

def set_bag_items(pyboy, items):
    write_byte(pyboy, BAG_ITEM_COUNT, len(items))
    for i, (item_name, qty) in enumerate(items):
        item_id = ITEM_IDS[item_name]
        addr = BAG_ITEMS_START + (i * 2)
        write_byte(pyboy, addr, item_id)
        write_byte(pyboy, addr + 1, qty)
    write_byte(pyboy, BAG_ITEMS_START + (len(items) * 2), 0xFF)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    BASE_DIR = Path(__file__).parent
    ROM_PATH = BASE_DIR / "PokemonRed.gb"
    
    # USAMOS UN ESTADO QUE YA ESTA EN EL GIMNASIO PARA EVITAR CORRUPCION
    INPUT_STATE = BASE_DIR / "PokemonCombatAgent" / "pewter_gym_valid.state"
    OUTPUT_STATE = BASE_DIR / "pewter_gym_configured.state"
    
    if not INPUT_STATE.exists():
        print(f"ERROR: No se encontro {INPUT_STATE}")
        exit(1)
        
    print(f"Base: {INPUT_STATE}")
    print(f"Destino: {OUTPUT_STATE}")
    
    pyboy = PyBoy(str(ROM_PATH), window="null", sound=False, sound_emulated=False)
    
    with open(INPUT_STATE, "rb") as f:
        pyboy.load_state(f)
    
    print("\nAplicando configuracion...")
    
    # 1. Equipo
    write_byte(pyboy, PARTY_SIZE, len(TEAM_CONFIG))
    write_byte(pyboy, PARTY_SPECIES_LIST + len(TEAM_CONFIG), 0xFF)
    for slot, conf in enumerate(TEAM_CONFIG):
        set_pokemon(pyboy, slot, conf)
        
    # 2. Mochila
    set_bag_items(pyboy, BAG_CONFIG)
    print("  Mochila actualizada")
    
    # 3. Dinero
    set_money(pyboy, MONEY)
    print(f"  Dinero: {MONEY}")
    
    # 4. Medallas
    write_byte(pyboy, BADGES_ADDR, BADGES)
    print("  Medallas reseteadas")
    
    # 5. Posicion (Asegurar)
    write_byte(pyboy, MAP_ID, 54)
    write_byte(pyboy, X_POS, 4)
    write_byte(pyboy, Y_POS, 13)
    print("  Posicion asegurada en Pewter Gym (54)")
    
    # Guardar
    with open(OUTPUT_STATE, "wb") as f:
        pyboy.save_state(f)
        
    pyboy.stop()
    print("\nEstado generado correctamente!")
