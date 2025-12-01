"""
Script interactivo para crear manualmente un estado de batalla.
Cargas el estado "corrupto", presionas A hasta llegar a batalla, y guardas.
"""

from pyboy import PyBoy
import sys

def create_manual_battle_state():
    """Crea un estado de batalla presionando A manualmente."""
    
    print("="*70)
    print("CREADOR MANUAL DE ESTADO DE BATALLA")
    print("="*70)
    print("\nCargando juego con ventana visible...")
    
    # Cargar PyBoy con ventana
    pyboy = PyBoy('PokemonRed.gb', window="SDL2")
    
    # Cargar estado inicial
    print("\nCargando estado base...")
    with open('has_pokedex_nballs.state', 'rb') as f:
        pyboy.load_state(f)
    
    print("\n" + "="*70)
    print("INSTRUCCIONES:")
    print("="*70)
    print("\n1. La ventana de PyBoy está abierta")
    print("2. Usa las FLECHAS del teclado para moverte")
    print("3. Presiona Z para el botón A")
    print("4. Presiona X para el botón B")
    print("5. Busca una batalla (camina en pasto alto)")
    print("6. Cuando estés EN EL MENÚ DE BATALLA, presiona:")
    print("   CTRL+C en esta terminal")
    print("\n" + "="*70)
    print("Esperando... (presiona CTRL+C cuando estés en batalla)")
    print("="*70 + "\n")
    
    try:
        # Loop infinito hasta que el usuario presione CTRL+C
        while True:
            pyboy.tick()
            
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("GUARDANDO ESTADO...")
        print("="*70)
        
        # Verificar si está en batalla
        in_battle = pyboy.memory[0xD057]
        
        if in_battle != 0:
            print(f"✓ En batalla detectada!")
            
            # Guardar estado
            output_file = 'manual_battle_state.state'
            with open(output_file, 'wb') as f:
                pyboy.save_state(f)
            
            # Guardar captura
            screenshot_file = 'manual_battle_screenshot.png'
            pyboy.screen.image.save(screenshot_file)
            
            print(f"\n✅ ÉXITO!")
            print(f"Estado guardado: {output_file}")
            print(f"Captura guardada: {screenshot_file}")
            print("\n" + "="*70)
            
        else:
            print("⚠️  No se detectó batalla activa.")
            print("El estado se guardará de todas formas...")
            
            output_file = 'manual_state.state'
            with open(output_file, 'wb') as f:
                pyboy.save_state(f)
            
            print(f"Estado guardado: {output_file}")
        
        pyboy.stop()
        print("\n✓ Proceso completado")

if __name__ == "__main__":
    create_manual_battle_state()
