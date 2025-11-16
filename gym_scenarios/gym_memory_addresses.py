"""
Direcciones de Memoria Extendidas para Pokémon Red - Escenarios de Gimnasios
=============================================================================

Direcciones RAM adicionales necesarias para configurar escenarios de gimnasios.
Basado en: https://datacrystal.tcrf.net/wiki/Pokémon_Red/Blue:RAM_map
"""

# ============================================================================
# DIRECCIONES DEL EQUIPO POKÉMON (Party)
# ============================================================================

PARTY_SIZE_ADDRESS = 0xD163  # Tamaño del equipo (1-6)
PARTY_ADDRESSES = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]  # Especies

# Estructura de cada Pokémon en el equipo (offsets de 0x2C = 44 bytes)
PARTY_DATA_START = 0xD16B

# Para cada Pokémon (índice 0-5), estructura de 44 bytes:
# Offset 0x00: Species
# Offset 0x01-0x02: Current HP (big endian)
# Offset 0x08: Level
# Offset 0x21: HP stat (max HP, 2 bytes)
# Offset 0x23: Attack stat (2 bytes)
# Offset 0x25: Defense stat (2 bytes)
# Offset 0x27: Speed stat (2 bytes)
# Offset 0x29: Special stat (2 bytes)

LEVELS_ADDRESSES = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
HP_ADDRESSES = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
MAX_HP_ADDRESSES = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]

# Movimientos (4 por Pokémon)
MOVES_PP_ADDRESSES = [
    [0xD188, 0xD189, 0xD18A, 0xD18B],  # Pokemon 1
    [0xD1B4, 0xD1B5, 0xD1B6, 0xD1B7],  # Pokemon 2
    [0xD1E0, 0xD1E1, 0xD1E2, 0xD1E3],  # Pokemon 3
    [0xD20C, 0xD20D, 0xD20E, 0xD20F],  # Pokemon 4
    [0xD238, 0xD239, 0xD23A, 0xD23B],  # Pokemon 5
    [0xD264, 0xD265, 0xD266, 0xD267],  # Pokemon 6
]

# ============================================================================
# DIRECCIONES DE POSICIÓN Y MAPA
# ============================================================================

X_POS_ADDRESS = 0xD362
Y_POS_ADDRESS = 0xD361
MAP_N_ADDRESS = 0xD35E

# Mapas de Gimnasios (IDs)
GYM_MAP_IDS = {
    'pewter': 54,      # Gimnasio de Brock (Pewter City)
    'cerulean': 65,    # Gimnasio de Misty (Cerulean City)
    'vermilion': 92,   # Gimnasio de Lt. Surge (Vermilion City)
    'celadon': 123,    # Gimnasio de Erika (Celadon City)
    'fuchsia': 146,    # Gimnasio de Koga (Fuchsia City)
    'saffron': 178,    # Gimnasio de Sabrina (Saffron City)
    'cinnabar': 166,   # Gimnasio de Blaine (Cinnabar Island)
    'viridian': 45,    # Gimnasio de Giovanni (Viridian City)
}

# ============================================================================
# DIRECCIONES DE ITEMS Y MOCHILA
# ============================================================================

BAG_ITEM_COUNT = 0xD31D  # Número de items en la mochila
BAG_ITEMS_START = 0xD31E  # Inicio de items (pares: item_id, cantidad)

# Items comunes útiles para gimnasios
ITEM_IDS = {
    'potion': 0x14,
    'super_potion': 0x15,
    'hyper_potion': 0x16,
    'max_potion': 0x17,
    'antidote': 0x18,
    'burn_heal': 0x19,
    'ice_heal': 0x1A,
    'awakening': 0x1B,
    'paralyze_heal': 0x1C,
    'full_restore': 0x1D,
    'full_heal': 0x1E,
    'revive': 0x1F,
    'max_revive': 0x20,
    'escape_rope': 0x28,
    'repel': 0x29,
    'poke_ball': 0x04,
    'great_ball': 0x05,
    'ultra_ball': 0x06,
    'x_attack': 0x31,
    'x_defend': 0x32,
    'x_speed': 0x33,
    'x_special': 0x34,
}

# ============================================================================
# DIRECCIONES DE MEDALLAS (BADGES)
# ============================================================================

BADGE_COUNT_ADDRESS = 0xD356  # Byte con bits de medallas

BADGE_BITS = {
    'boulder': 0,    # Brock (Pewter)
    'cascade': 1,    # Misty (Cerulean)
    'thunder': 2,    # Lt. Surge (Vermilion)
    'rainbow': 3,    # Erika (Celadon)
    'soul': 4,       # Koga (Fuchsia)
    'marsh': 5,      # Sabrina (Saffron)
    'volcano': 6,    # Blaine (Cinnabar)
    'earth': 7,      # Giovanni (Viridian)
}

# ============================================================================
# DIRECCIONES DE DINERO
# ============================================================================

MONEY_ADDRESS_1 = 0xD347  # BCD
MONEY_ADDRESS_2 = 0xD348  # BCD
MONEY_ADDRESS_3 = 0xD349  # BCD

# ============================================================================
# DIRECCIONES DE COMBATE
# ============================================================================

BATTLE_TYPE = 0xD057  # 0=wild, 1=trainer, 2=old man tutorial
IN_BATTLE = 0xD057    # Non-zero cuando está en combate

OPPONENT_LEVELS_ADDRESSES = [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]

# ============================================================================
# POKÉMON IDS (Index Numbers)
# ============================================================================

POKEMON_IDS = {
    # Starters
    'bulbasaur': 0x99,
    'ivysaur': 0x09,
    'venusaur': 0x9A,
    'charmander': 0xB0,
    'charmeleon': 0xB2,
    'charizard': 0xB4,
    'squirtle': 0xB1,
    'wartortle': 0xB3,
    'blastoise': 0x1C,
    
    # Comunes early game
    'pidgey': 0x24,
    'pidgeotto': 0x96,
    'pidgeot': 0x97,
    'rattata': 0xA5,
    'raticate': 0xA6,
    'spearow': 0x05,
    'fearow': 0x23,
    'caterpie': 0x7B,
    'metapod': 0x7C,
    'butterfree': 0x7D,
    'weedle': 0x70,
    'kakuna': 0x71,
    'beedrill': 0x72,
    'pikachu': 0x54,
    'raichu': 0x55,
    'sandshrew': 0x60,
    'sandslash': 0x61,
    'nidoran_f': 0x0F,
    'nidorina': 0xA8,
    'nidoqueen': 0x10,
    'nidoran_m': 0x03,
    'nidorino': 0xA7,
    'nidoking': 0x07,
    'clefairy': 0x04,
    'clefable': 0x8E,
    'jigglypuff': 0x64,
    'wigglytuff': 0x65,
    'zubat': 0x6B,
    'golbat': 0x82,
    'oddish': 0xB9,
    'gloom': 0xBA,
    'vileplume': 0xBB,
    'paras': 0x6D,
    'parasect': 0x2E,
    'venonat': 0x41,
    'venomoth': 0x77,
    'diglett': 0x3B,
    'dugtrio': 0x76,
    'meowth': 0x4D,
    'persian': 0x90,
    'psyduck': 0x2F,
    'golduck': 0x80,
    'mankey': 0x39,
    'primeape': 0x75,
    'growlithe': 0x21,
    'arcanine': 0x14,
    'poliwag': 0x47,
    'poliwhirl': 0x6E,
    'poliwrath': 0x6F,
    'abra': 0x94,
    'kadabra': 0x26,
    'alakazam': 0x95,
    'machop': 0x6A,
    'machoke': 0x29,
    'machamp': 0x7E,
    'bellsprout': 0xBC,
    'weepinbell': 0xBD,
    'victreebel': 0xBE,
    'tentacool': 0x18,
    'tentacruel': 0x9B,
    'geodude': 0xA9,
    'graveler': 0x27,
    'golem': 0x31,
    'ponyta': 0xA3,
    'rapidash': 0xA4,
    'slowpoke': 0x25,
    'slowbro': 0x08,
    'magnemite': 0xAD,
    'magneton': 0x36,
    'farfetchd': 0x40,
    'doduo': 0x46,
    'dodrio': 0x74,
    'seel': 0x3A,
    'dewgong': 0x78,
    'grimer': 0x0D,
    'muk': 0x88,
    'shellder': 0x17,
    'cloyster': 0x8B,
    'gastly': 0x19,
    'haunter': 0x93,
    'gengar': 0x0E,
    'onix': 0x22,
    'drowzee': 0x30,
    'hypno': 0x81,
    'krabby': 0x4E,
    'kingler': 0x8A,
    'voltorb': 0x06,
    'electrode': 0x8D,
    'exeggcute': 0x0C,
    'exeggutor': 0x0A,
    'cubone': 0x11,
    'marowak': 0x91,
    'hitmonlee': 0x2B,
    'hitmonchan': 0x2C,
    'lickitung': 0x0B,
    'koffing': 0x37,
    'weezing': 0x8F,
    'rhyhorn': 0x12,
    'rhydon': 0x01,
    'chansey': 0x28,
    'tangela': 0x1E,
    'kangaskhan': 0x02,
    'horsea': 0x5C,
    'seadra': 0x5D,
    'goldeen': 0x9D,
    'seaking': 0x9E,
    'staryu': 0x1B,
    'starmie': 0x98,
    'scyther': 0x1A,
    'jynx': 0x48,
    'electabuzz': 0x35,
    'magmar': 0x33,
    'pinsir': 0x1D,
    'tauros': 0x3C,
    'magikarp': 0x85,
    'gyarados': 0x16,
    'lapras': 0x13,
    'ditto': 0x4C,
    'eevee': 0x66,
    'vaporeon': 0x69,
    'jolteon': 0x68,
    'flareon': 0x67,
}

# ============================================================================
# EVENT FLAGS (para verificar progreso)
# ============================================================================

EVENT_FLAGS_START_ADDRESS = 0xD747
EVENT_FLAGS_END_ADDRESS = 0xD886

# Eventos específicos de gimnasios (offset desde EVENT_FLAGS_START)
GYM_EVENTS = {
    'brock_defeated': (0xD755, 7),      # Bit 7 en 0xD755
    'misty_defeated': (0xD755, 6),      # Bit 6 en 0xD755
    'surge_defeated': (0xD773, 0),      # Bit 0 en 0xD773
    'erika_defeated': (0xD773, 1),      # Bit 1 en 0xD773
    'koga_defeated': (0xD792, 7),       # Bit 7 en 0xD792
    'sabrina_defeated': (0xD7D4, 7),    # Bit 7 en 0xD7D4
    'blaine_defeated': (0xD7EE, 7),     # Bit 7 en 0xD7EE
    'giovanni_defeated': (0xD751, 7),   # Bit 7 en 0xD751 (último)
}
