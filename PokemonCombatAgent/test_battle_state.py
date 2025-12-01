"""
Test script to verify battle states are valid and usable
"""
from pyboy import PyBoy
import time

def test_state(state_file):
    print(f"\n{'='*60}")
    print(f"Testing: {state_file}")
    print('='*60)
    
    pyboy = PyBoy(
        'PokemonRed.gb',
        window='SDL2',
        sound=False,
        sound_emulated=False
    )
    
    # Load state
    try:
        with open(state_file, 'rb') as f:
            pyboy.load_state(f)
        print("✓ State loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load state: {e}")
        pyboy.stop()
        return
    
    # Run for a few frames
    print("\nRunning 60 frames...")
    for i in range(60):
        pyboy.tick()
        if i % 10 == 0:
            print(f"  Frame {i}/60")
    
    # Check memory for battle state
    BATTLE_TYPE = 0xD057
    IN_BATTLE = 0xD05A
    PLAYER_HP_HIGH = 0xD16C
    ENEMY_HP_HIGH = 0xCFE6
    
    battle_type = pyboy.memory[BATTLE_TYPE]
    in_battle = pyboy.memory[IN_BATTLE]
    player_hp = (pyboy.memory[PLAYER_HP_HIGH] << 8) | pyboy.memory[PLAYER_HP_HIGH + 1]
    enemy_hp = (pyboy.memory[ENEMY_HP_HIGH] << 8) | pyboy.memory[ENEMY_HP_HIGH + 1]
    
    print(f"\nMemory Check:")
    print(f"  Battle Type (0xD057): {battle_type} (0=none, 1=wild, 2=trainer)")
    print(f"  In Battle Flag (0xD05A): {in_battle}")
    print(f"  Player HP: {player_hp}")
    print(f"  Enemy HP: {enemy_hp}")
    
    if battle_type > 0 or in_battle > 0 or enemy_hp > 0:
        print("\n✓ Appears to be IN BATTLE")
    else:
        print("\n⚠ NOT in battle - may need initialization")
    
    # Let it run for 5 seconds so you can see
    print("\nDisplaying for 5 seconds... (press Ctrl+C to skip)")
    try:
        for _ in range(300):  # 5 seconds at 60 fps
            pyboy.tick()
            time.sleep(1/60)
    except KeyboardInterrupt:
        print("Skipped")
    
    pyboy.stop()
    print("✓ Test complete\n")

if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    battle_states = sorted(Path('battle_states').glob('*_battle.state'))
    
    if len(sys.argv) > 1:
        # Test specific state
        test_state(sys.argv[1])
    else:
        # Test all states
        print(f"Found {len(battle_states)} battle states\n")
        for state in battle_states:
            test_state(str(state))
            
        print("\n" + "="*60)
        print("All tests complete!")
        print("="*60)
