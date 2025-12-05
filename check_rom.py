
from pyboy import PyBoy

rom_path = 'PokemonRed.gb'
with open(rom_path, 'rb') as f:
    rom_data = f.read()

title = rom_data[0x134:0x143].decode('ascii', errors='ignore')
version = rom_data[0x14C]
global_checksum = (rom_data[0x14E] << 8) | rom_data[0x14F]

print(f"ROM Title: {title}")
print(f"Version: {version}")
print(f"Global Checksum: {hex(global_checksum)}")

# Check state file header if possible (PyBoy states are zips or custom)
# We can just try to load it in PyBoy and print metadata if any
