"""
Generador de Archivos .state para Escenarios de Gimnasios
=========================================================

Este script crea archivos de estado (.state) de PyBoy para cada uno de los 8 gimnasios,
configurando el equipo Pokémon, items, dinero, medallas y posición según las especificaciones.

Uso:
    python generate_gym_states.py

Requisitos:
    - PyBoy instalado
    - PokemonRed.gb en el directorio raíz
    - Archivos team_config.json en cada carpeta de gimnasio
"""

import json
import os
from pathlib import Path
from pyboy import PyBoy
import sys

# Agregar path de baselines para importar memory_addresses
sys.path.append(str(Path(__file__).parent.parent / "baselines"))
sys.path.append(str(Path(__file__).parent))

from gym_memory_addresses import *


class GymStateGenerator:
    """Genera archivos .state para escenarios de gimnasios"""
    
    def __init__(self, rom_path):
        self.rom_path = rom_path
        self.pyboy = None
        
    def init_pyboy(self):
        """Inicializa PyBoy en modo headless"""
        if self.pyboy:
            self.pyboy.stop()
        
        # PyBoy v2+ usa los argumentos 'window' y 'sound' (los anteriores fueron deprecados)
        self.pyboy = PyBoy(
            self.rom_path,
            window="null",  # Ejecuta sin mostrar ventana
            sound=False
        )
        print(f"✓ PyBoy inicializado con ROM: {self.rom_path}")
    
    def write_memory(self, address, value):
        """Escribe un valor en una dirección de memoria"""
        if hasattr(self.pyboy, "set_memory_value"):
            self.pyboy.set_memory_value(address, value)
        else:
            # PyBoy v2 expone la memoria como un buffer mutable
            self.pyboy.memory[address] = value & 0xFF
    
    def write_word(self, address, value):
        """Escribe un valor de 16-bit (word) en big-endian"""
        high_byte = (value >> 8) & 0xFF
        low_byte = value & 0xFF
        self.write_memory(address, high_byte)
        self.write_memory(address + 1, low_byte)
    
    def write_bcd(self, value):
        """Convierte un número decimal a BCD (Binary Coded Decimal)"""
        return ((value // 10) << 4) | (value % 10)
    
    def set_position(self, x, y, map_id):
        """Establece la posición del jugador en el mapa"""
        self.write_memory(X_POS_ADDRESS, x)
        self.write_memory(Y_POS_ADDRESS, y)
        self.write_memory(MAP_N_ADDRESS, map_id)
        print(f"  - Posición: X={x}, Y={y}, Mapa={map_id}")
    
    def set_badges(self, badge_bits):
        """Establece las medallas obtenidas"""
        self.write_memory(BADGE_COUNT_ADDRESS, badge_bits)
        badge_count = bin(badge_bits).count('1')
        print(f"  - Medallas: {badge_count} ({bin(badge_bits)})")
    
    def set_money(self, amount):
        """Establece el dinero del jugador en formato BCD"""
        # Pokémon Red usa BCD para dinero (3 bytes)
        hundreds_thousands = amount // 10000
        thousands = (amount // 100) % 100
        ones = amount % 100
        
        self.write_memory(MONEY_ADDRESS_1, self.write_bcd(hundreds_thousands))
        self.write_memory(MONEY_ADDRESS_2, self.write_bcd(thousands))
        self.write_memory(MONEY_ADDRESS_3, self.write_bcd(ones))
        print(f"  - Dinero: ${amount}")
    
    def set_party_size(self, size):
        """Establece el tamaño del equipo"""
        self.write_memory(PARTY_SIZE_ADDRESS, size)
        print(f"  - Tamaño del equipo: {size} Pokémon")
    
    def set_pokemon(self, slot, pokemon_data):
        """
        Configura un Pokémon en el equipo
        
        Args:
            slot: Posición en el equipo (0-5)
            pokemon_data: Diccionario con datos del Pokémon
        """
        if slot < 0 or slot > 5:
            print(f"  ⚠ Slot inválido: {slot}")
            return
        
        # Especie
        species_id = pokemon_data.get('species_id', 0)
        self.write_memory(PARTY_ADDRESSES[slot], species_id)
        
        # Nivel
        level = pokemon_data.get('level', 5)
        self.write_memory(LEVELS_ADDRESSES[slot], level)
        
        # HP actual
        current_hp = pokemon_data.get('current_hp', 20)
        self.write_word(HP_ADDRESSES[slot], current_hp)
        
        # HP máximo
        max_hp = pokemon_data.get('max_hp', 20)
        self.write_word(MAX_HP_ADDRESSES[slot], max_hp)
        
        species_name = pokemon_data.get('species', 'Unknown')
        print(f"  - Slot {slot + 1}: {species_name} Lv.{level} ({current_hp}/{max_hp} HP)")
    
    def set_bag_items(self, items):
        """
        Configura los items en la mochila
        
        Args:
            items: Lista de diccionarios con 'item_id' y 'quantity'
        """
        # Número de items
        item_count = min(len(items), 20)  # Máximo 20 items
        self.write_memory(BAG_ITEM_COUNT, item_count)
        
        # Escribir cada item (formato: item_id, quantity)
        for i, item in enumerate(items[:20]):
            base_addr = BAG_ITEMS_START + (i * 2)
            item_id = item.get('item_id', 0)
            quantity = item.get('quantity', 1)
            
            self.write_memory(base_addr, item_id)
            self.write_memory(base_addr + 1, quantity)
            
            item_name = item.get('item', 'Unknown')
            print(f"  - Item: {item_name} x{quantity}")
        
        # Terminator
        self.write_memory(BAG_ITEMS_START + (item_count * 2), 0xFF)
    
    def load_base_state(self, base_state_path):
        """Carga un estado base desde archivo"""
        if os.path.exists(base_state_path):
            with open(base_state_path, "rb") as f:
                self.pyboy.load_state(f)
            print(f"✓ Estado base cargado: {base_state_path}")
            return True
        else:
            print(f"⚠ Estado base no encontrado: {base_state_path}")
            return False
    
    def generate_gym_state(self, gym_folder, output_filename="gym_scenario.state"):
        """
        Genera un archivo .state para un gimnasio específico
        
        Args:
            gym_folder: Path a la carpeta del gimnasio
            output_filename: Nombre del archivo de salida
        """
        gym_folder = Path(gym_folder)
        config_path = gym_folder / "team_config.json"
        
        if not config_path.exists():
            print(f"✗ Config no encontrado: {config_path}")
            return False
        
        # Cargar configuración
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        gym_name = config.get('gym_name', 'Unknown Gym')
        print(f"\n{'='*60}")
        print(f"Generando estado para: {gym_name}")
        print(f"{'='*60}")
        
        # Reiniciar PyBoy con estado limpio
        self.init_pyboy()
        
        # Cargar estado base (si existe)
        base_state = Path(__file__).parent.parent / "init.state"
        if not self.load_base_state(base_state):
            # Si no hay estado base, avanzar algunos frames
            for _ in range(100):
                self.pyboy.tick()
        
        # Configurar equipo
        team = config.get('player_team', [])
        self.set_party_size(len(team))
        
        for pokemon_data in team:
            slot = pokemon_data.get('slot', 1) - 1  # Convertir a 0-indexed
            self.set_pokemon(slot, pokemon_data)
        
        # Configurar items
        items = config.get('bag_items', [])
        if items:
            print(f"\n  Configurando {len(items)} items:")
            self.set_bag_items(items)
        
        # Configurar dinero
        money = config.get('money', 0)
        self.set_money(money)
        
        # Configurar medallas
        badge_bits = config.get('badge_bits', 0)
        self.set_badges(badge_bits)
        
        # Configurar posición
        start_pos = config.get('start_position', {'x': 4, 'y': 13})
        map_id = config.get('map_id', 0)
        self.set_position(start_pos['x'], start_pos['y'], map_id)
        
        # Guardar estado
        output_path = gym_folder / output_filename
        with open(output_path, "wb") as f:
            self.pyboy.save_state(f)
        
        print(f"\n✓ Estado guardado: {output_path}")
        print(f"{'='*60}\n")
        
        return True
    
    def generate_all_gym_states(self, gym_scenarios_folder):
        """Genera estados para todos los gimnasios"""
        gym_scenarios_folder = Path(gym_scenarios_folder)
        
        # Buscar todas las carpetas de gimnasios
        gym_folders = sorted([
            folder for folder in gym_scenarios_folder.iterdir()
            if folder.is_dir() and folder.name.startswith('gym')
        ])
        
        print(f"\n{'*'*60}")
        print(f"GENERADOR DE ESTADOS PARA GIMNASIOS POKÉMON")
        print(f"{'*'*60}")
        print(f"Gimnasios encontrados: {len(gym_folders)}\n")
        
        success_count = 0
        for gym_folder in gym_folders:
            if self.generate_gym_state(gym_folder):
                success_count += 1
        
        print(f"\n{'*'*60}")
        print(f"RESUMEN: {success_count}/{len(gym_folders)} estados generados exitosamente")
        print(f"{'*'*60}\n")
        
        if self.pyboy:
            self.pyboy.stop()


def main():
    """Función principal"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Path al ROM
    rom_path = project_root / "PokemonRed.gb"
    
    if not rom_path.exists():
        print(f"✗ ROM no encontrado: {rom_path}")
        print("  Por favor, coloca PokemonRed.gb en el directorio raíz del proyecto")
        return
    
    # Path a la carpeta de escenarios
    gym_scenarios_folder = script_dir
    
    # Crear generador y procesar todos los gimnasios
    generator = GymStateGenerator(str(rom_path))
    generator.generate_all_gym_states(gym_scenarios_folder)


if __name__ == "__main__":
    main()
