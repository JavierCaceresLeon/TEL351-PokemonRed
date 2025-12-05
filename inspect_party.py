
from pyboy import PyBoy
import os

STATE_FILE = 'pewter_gym_configured.state'
ROM_FILE = 'PokemonRed.gb'

pyboy = PyBoy(ROM_FILE, window='null', sound=False, sound_emulated=False)
with open(STATE_FILE, 'rb') as f:
    pyboy.load_state(f)

print(f"Loaded {STATE_FILE}")

print(f"Party Size (0xD163): {pyboy.memory[0xD163]}")
print(f"Species List (0xD164-9): {[hex(pyboy.memory[0xD164+i]) for i in range(6)]}")
print(f"Terminator (0xD164+Size): {hex(pyboy.memory[0xD164 + pyboy.memory[0xD163]])}")

# Check Pokemon 2 (Slot 1)
base_addr = 0xD16B + 44
print(f"\nPokemon 2 (Slot 1) @ {hex(base_addr)}:")
print(f"  Species (0x00): {hex(pyboy.memory[base_addr])}")
print(f"  HP (0x01-02): {pyboy.memory[base_addr+1]*256 + pyboy.memory[base_addr+2]}")
print(f"  Level (0x21): {pyboy.memory[base_addr+0x21]}")

# Check Nicknames
print("\nNicknames (0xD2B5):")
for i in range(3):
    addr = 0xD2B5 + (i * 11)
    raw_bytes = [pyboy.memory[addr+j] for j in range(11)]
    print(f"  Slot {i}: {raw_bytes}")

# Check OT Names
print("\nOT Names (0xD273):")
for i in range(3):
    addr = 0xD273 + (i * 11)
    raw_bytes = [pyboy.memory[addr+j] for j in range(11)]
    print(f"  Slot {i}: {raw_bytes}")

pyboy.stop()
