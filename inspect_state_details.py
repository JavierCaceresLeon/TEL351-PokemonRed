from pyboy import PyBoy
from pathlib import Path

ROM = "PokemonRed.gb"
STATE = "has_pokedex_nballs.state"

py = PyBoy(ROM, window="null", sound=False, sound_emulated=False)
with open(STATE, "rb") as f:
    py.load_state(f)

print("Party size:", py.memory[0xD163])
for slot in range(py.memory[0xD163]):
    base = 0xD16B + slot * 0x2C
    species = py.memory[0xD164 + slot]
    level = py.memory[base + 0x21]
    hp = (py.memory[base + 1] << 8) | py.memory[base + 2]
    moves = [py.memory[base + 0x08 + i] for i in range(4)]
    print(f" Slot {slot+1}: species={hex(species)}, level={level}, HP={hp}, moves={moves}")

bag_count = py.memory[0xD31D]
print("Bag count:", bag_count)
for i in range(bag_count):
    item = py.memory[0xD31E + i * 2]
    qty = py.memory[0xD31F + i * 2]
    print(f" Item {i+1}: id={hex(item)}, qty={qty}")

money_bytes = [py.memory[0xD347 + i] for i in range(3)]
print("Money bytes:", [hex(b) for b in money_bytes])
print("Badges bits:", bin(py.memory[0xD356]))
print("Map ID:", py.memory[0xD35E])
print("Position (x,y):", py.memory[0xD362], py.memory[0xD361])

py.stop()
