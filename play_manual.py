
import sys
import os
from pathlib import Path
from pyboy import PyBoy

# Configuración
ROM_PATH = 'PokemonRed.gb'
START_STATE = 'has_pokedex_nballs.state' # Punto de partida recomendado
OUTPUT_STATE = 'pewter_gym_manual.state' # Nombre del archivo que se generará

def main():
    if not os.path.exists(ROM_PATH):
        print(f"Error: No se encuentra {ROM_PATH}")
        return

    print("="*50)
    print("MODO MANUAL DE POKEMON RED")
    print("="*50)
    print(f"1. Se abrirá la ventana del juego.")
    print(f"2. Juega usando el teclado (Flechas, A=Z, B=X, Start=Enter, Select=Backspace).")
    print(f"3. Mueve al personaje hasta el Gimnasio de Brock.")
    print(f"4. Cuando estés listo para guardar, vuelve a esta terminal.")
    print(f"5. Presiona CTRL+C para guardar el estado y salir.")
    print("="*50)

    # Inicializar PyBoy
    pyboy = PyBoy(ROM_PATH, window='SDL2', sound=False)
    
    # Cargar estado inicial si existe
    if os.path.exists(START_STATE):
        print(f"Cargando estado inicial: {START_STATE}")
        with open(START_STATE, 'rb') as f:
            pyboy.load_state(f)
    else:
        print(f"Advertencia: No se encontró {START_STATE}, iniciando desde cero.")

    pyboy.set_emulation_speed(1)
    print("\nJuego iniciado. Presiona CTRL+C en la terminal para guardar y salir.")

    try:
        while True:
            if not pyboy.tick():
                break # Ventana cerrada
    except KeyboardInterrupt:
        print("\n\n¡Detenido por el usuario!")
    
    # Guardar estado
    print(f"Guardando estado en: {OUTPUT_STATE}...")
    with open(OUTPUT_STATE, 'wb') as f:
        pyboy.save_state(f)
    print("¡Guardado exitoso!")
    
    pyboy.stop()

if __name__ == '__main__':
    main()
