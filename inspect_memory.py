
from pyboy import PyBoy
import os

STATE_FILE = 'has_pokedex_nballs.state'
ROM_FILE = 'PokemonRed.gb'

if not os.path.exists(STATE_FILE):
    print(f"Error: {STATE_FILE} not found")
    exit(1)

pyboy = PyBoy(ROM_FILE, window='null', sound=False, sound_emulated=False)
with open(STATE_FILE, 'rb') as f:
    pyboy.load_state(f)

print(f"Loaded {STATE_FILE}")

# Check Standard Addresses
print("\n--- STANDARD ADDRESSES ---")
print(f"Party Count (0xD163): {pyboy.memory[0xD163]}")
print(f"Species 1 (0xD164): {hex(pyboy.memory[0xD164])}")
print(f"Bag Count (0xD31D): {pyboy.memory[0xD31D]}")
print(f"Money (0xD347-9): {pyboy.memory[0xD347]:02x}{pyboy.memory[0xD348]:02x}{pyboy.memory[0xD349]:02x}")
print(f"Map ID (0xD35E): {pyboy.memory[0xD35E]}")
print(f"X (0xD362): {pyboy.memory[0xD362]}")
print(f"Y (0xD361): {pyboy.memory[0xD361]}")

# Check User Research Addresses
print("\n--- USER RESEARCH ADDRESSES ---")
print(f"Bag Items? (0xD1C0): {pyboy.memory[0xD1C0]}")
print(f"Money? (0xD34B): {pyboy.memory[0xD34B]}")
print(f"Team? (0xD164): {hex(pyboy.memory[0xD164])}") # Same as standard species list

# Check Player Name (Standard 0xD158)
name = ""
for i in range(11):
    val = pyboy.memory[0xD158 + i]
    if val == 0x50: break # Terminator
    name += chr(val) # This won't be ASCII, but let's see bytes
print(f"Player Name Bytes (0xD158): {[pyboy.memory[0xD158+i] for i in range(11)]}")

pyboy.stop()
