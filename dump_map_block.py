import sys
from pathlib import Path
from pyboy import PyBoy

if len(sys.argv) < 2:
    print("Usage: python dump_map_block.py <state>")
    sys.exit(1)

rom = "PokemonRed.gb"
state = Path(sys.argv[1])
if not state.exists():
    print(f"State {state} not found")
    sys.exit(1)

py = PyBoy(rom, window="null", sound=False, sound_emulated=False)
with open(state, "rb") as f:
    py.load_state(f)

print("State:", state)
print("Party size:", py.memory[0xD163])
print("Map ID:", py.memory[0xD35E])
print("Coords (X,Y):", py.memory[0xD362], py.memory[0xD361])
print("Facing Dir:", hex(py.memory[0xD364]))
print("Map header bytes (0xD35E-0xD37E):")
print([hex(py.memory[0xD35E + i]) for i in range(0x21)])
py.stop()
