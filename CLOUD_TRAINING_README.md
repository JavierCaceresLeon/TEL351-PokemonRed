# Guía de Entrenamiento en la Nube (Google Colab)

Esta guía te permitirá entrenar tus agentes de Pokémon Red en Google Colab, visualizando el progreso en tiempo real.

## Pasos Previos

1.  **Preparar Archivos:**
    *   Asegúrate de tener el archivo `PokemonRed.gb` en la carpeta `TEL351-PokemonRed`.
    *   Comprime la carpeta `TEL351-PokemonRed` completa en un archivo `.zip` (ej. `TEL351-PokemonRed.zip`).

2.  **Google Drive:**
    *   Sube el archivo `TEL351-PokemonRed.zip` a la raíz de tu Google Drive.

3.  **Google Colab:**
    *   Abre [Google Colab](https://colab.research.google.com/).
    *   Sube el archivo `Colab_Train.ipynb` (que se ha generado en tu espacio de trabajo) a Colab.
    *   Sigue las instrucciones dentro del Notebook.

## Contenido del Notebook

El notebook realizará lo siguiente:
1.  Montará tu Google Drive.
2.  Descomprimirá el proyecto.
3.  Instalará las dependencias necesarias (PyBoy, Stable Baselines 3, etc.).
4.  Configurará el entorno de visualización en vivo.
5.  Ejecutará el entrenamiento del agente seleccionado en el escenario elegido.

## Visualización

El notebook incluye un sistema de visualización que muestra la pantalla del GameBoy actualizándose en tiempo real (con un ligero retraso para no afectar el rendimiento) dentro de la celda de salida de Colab.
