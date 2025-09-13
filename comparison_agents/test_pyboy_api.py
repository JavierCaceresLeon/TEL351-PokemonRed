#!/usr/bin/env python3
"""
Script de prueba para verificar la API de PyBoy
"""
import sys
import os

# Agregar el directorio v2 al path
v2_path = os.path.join(os.path.dirname(__file__), "..", "v2")
sys.path.insert(0, v2_path)

try:
    from pyboy import PyBoy
    
    print("Testing PyBoy API...")
    
    # Path al ROM
    rom_path = "../PokemonRed.gb"
    
    # Intentar crear PyBoy
    pyboy = PyBoy(
        rom_path,
        window="SDL2",
        debug=False,
    )
    
    print("PyBoy creado exitosamente")
    
    # Verificar qué métodos están disponibles
    print("Métodos disponibles en PyBoy:")
    methods = [method for method in dir(pyboy) if not method.startswith('_')]
    for method in sorted(methods):  # Mostrar todos los métodos
        print(f"  - {method}")
    
    # Verificar pantalla
    if hasattr(pyboy, 'screen'):
        print("✅ screen directamente disponible")
        screen = pyboy.screen
        print("Métodos disponibles en screen:")
        screen_methods = [method for method in dir(screen) if not method.startswith('_')]
        for method in sorted(screen_methods):
            print(f"  - {method}")
    else:
        print("❌ No se encontró manera de acceder a screen")
    
    # Verificar memoria
    if hasattr(pyboy, 'get_memory_value'):
        print("✅ get_memory_value disponible")
    elif hasattr(pyboy, 'memory'):
        print("✅ memory disponible")
        # Intentar leer una dirección de memoria
        try:
            value = pyboy.memory[0x0000]
            print(f"✅ Memory read successful: {value}")
        except Exception as e:
            print(f"❌ Memory read failed: {e}")
    else:
        print("❌ No se encontró manera de acceder a memoria")
    
    pyboy.stop()
    print("Test completado")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()