"""
Crea un archivo .state funcional para batalla Pokemon.
Este script:
1. Carga el juego desde has_pokedex_nballs.state
2. Navega hasta una batalla salvaje
3. Guarda el estado DURANTE la batalla activa (en el menú de batalla)
"""

from pyboy import PyBoy
import time

def wait_frames(pyboy, num_frames):
    """Espera un número específico de frames."""
    for _ in range(num_frames):
        pyboy.tick()

def press_button(pyboy, button, frames=24):
    """Presiona un botón por un número de frames."""
    pyboy.button_press(button)
    wait_frames(pyboy, frames)
    pyboy.button_release(button)
    wait_frames(pyboy, frames)

def is_in_battle(pyboy):
    """Verifica si estamos en batalla."""
    return pyboy.memory[0xD057] != 0

def is_in_battle_menu(pyboy):
    """Verifica si estamos en el menú de batalla (no en texto)."""
    in_battle = is_in_battle(pyboy)
    text_active = pyboy.memory[0xCC57]
    return in_battle and text_active == 0

def create_battle_state():
    """Crea un estado de batalla funcional."""
    
    print("="*70)
    print("CREACIÓN DE ESTADO DE BATALLA FUNCIONAL")
    print("="*70)
    
    # Cargar el juego
    print("\n1. Cargando PokemonRed.gb...")
    pyboy = PyBoy('PokemonRed.gb', window="null")
    
    # Cargar estado inicial con Pokedex y Pokeballs
    print("2. Cargando estado inicial (has_pokedex_nballs.state)...")
    with open('has_pokedex_nballs.state', 'rb') as f:
        pyboy.load_state(f)
    
    wait_frames(pyboy, 60)
    
    # Verificar que no estamos en batalla
    if is_in_battle(pyboy):
        print("   ⚠️  Ya estamos en batalla, saliendo...")
        # Presionar RUN varias veces
        for _ in range(5):
            press_button(pyboy, "a")
            press_button(pyboy, "down")
            press_button(pyboy, "a")
            wait_frames(pyboy, 120)
            if not is_in_battle(pyboy):
                break
    
    print("3. Buscando batalla salvaje...")
    print("   Caminando en pasto alto...")
    
    # Caminar en círculos hasta encontrar batalla
    max_attempts = 500
    for attempt in range(max_attempts):
        # Caminar en patrón circular
        direction = ["up", "right", "down", "left"][attempt % 4]
        press_button(pyboy, direction, frames=12)
        
        wait_frames(pyboy, 30)
        
        if is_in_battle(pyboy):
            print(f"\n   ✓ ¡Batalla encontrada después de {attempt} pasos!")
            break
            
        if attempt % 10 == 0:
            print(f"   Intentos: {attempt}/{max_attempts}")
    
    if not is_in_battle(pyboy):
        print("\n   ❌ No se encontró batalla después de", max_attempts, "intentos")
        pyboy.stop()
        return False
    
    # Esperar a que aparezca el menú de batalla
    print("\n4. Esperando al menú de batalla...")
    wait_frames(pyboy, 60)
    
    # Avanzar diálogo inicial
    for _ in range(10):
        press_button(pyboy, "a", frames=12)
        wait_frames(pyboy, 30)
        
        if is_in_battle_menu(pyboy):
            print("   ✓ Menú de batalla alcanzado!")
            break
    
    wait_frames(pyboy, 60)
    
    # Verificar estado final
    print("\n5. Verificando estado de memoria...")
    memory = pyboy.memory
    
    in_battle = memory[0xD057]
    battle_type = memory[0xD05A]
    
    # HP del jugador
    player_hp = (memory[0xD016] << 8) | memory[0xD017]
    player_max_hp = (memory[0xD018] << 8) | memory[0xD019]
    player_level = memory[0xD01C]
    
    # HP del enemigo  
    enemy_hp = (memory[0xCFE7] << 8) | memory[0xCFE8]
    enemy_max_hp = (memory[0xCFE9] << 8) | memory[0xCFEA]
    enemy_level = memory[0xCFF3]
    enemy_species = memory[0xCFE5]
    
    print(f"   En batalla: {in_battle != 0}")
    print(f"   Tipo batalla: {battle_type}")
    print(f"   HP Jugador: {player_hp}/{player_max_hp} (Nivel {player_level})")
    print(f"   HP Enemigo: {enemy_hp}/{enemy_max_hp} (Nivel {enemy_level}, Especie {enemy_species})")
    
    # Validar que los valores tienen sentido
    if player_hp > 500 or player_max_hp > 500 or enemy_hp > 500:
        print("\n   ❌ ERROR: Valores de HP parecen corruptos")
        print("   Esto podría ser un problema con las direcciones de memoria")
        pyboy.stop()
        return False
    
    # Guardar estado
    output_file = 'generated_battle_states/functional_battle.state'
    print(f"\n6. Guardando estado en: {output_file}")
    
    with open(output_file, 'wb') as f:
        pyboy.save_state(f)
    
    # Guardar captura de pantalla
    screenshot_file = 'generated_battle_states/functional_battle_screenshot.png'
    pyboy.screen.image.save(screenshot_file)
    print(f"   ✓ Captura guardada: {screenshot_file}")
    
    pyboy.stop()
    
    print("\n" + "="*70)
    print("✓ ESTADO DE BATALLA CREADO EXITOSAMENTE")
    print("="*70)
    print(f"\nArchivo: {output_file}")
    print(f"Estado: En batalla activa, menú de batalla visible")
    print(f"Listo para usar en entrenamiento y comparación")
    print("="*70 + "\n")
    
    return True

if __name__ == "__main__":
    success = create_battle_state()
    
    if not success:
        print("\n⚠️  ADVERTENCIA: No se pudo crear el estado de batalla")
        print("Posibles soluciones:")
        print("1. Verificar que has_pokedex_nballs.state existe")
        print("2. Aumentar max_attempts para buscar batalla más tiempo")
        print("3. Verificar direcciones de memoria en el código")
