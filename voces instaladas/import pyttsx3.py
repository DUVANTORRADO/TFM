import pyttsx3

# Inicializar el motor de texto a voz
engine = pyttsx3.init()

# Obtener las voces disponibles
voices = engine.getProperty('voices')

# Mostrar las voces instaladas
for index, voice in enumerate(voices):
    print(f"Voz {index + 1}:")
    print(f" - ID: {voice.id}")
    print(f" - Nombre: {voice.name}")
    print(f" - Lenguaje: {voice.languages}")
    print()