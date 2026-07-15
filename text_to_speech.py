import pyttsx3

def text_to_speech(text, gender="female", language="es"):
    """
    Convierte texto a voz utilizando el motor pyttsx3.
    :param text: Texto a convertir en voz.
    :param gender: Género de la voz ("male" o "female").
    :param language: Idioma de la voz ("es" para español, "en" para inglés).
    """
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')  # Obtiene la lista de voces disponibles en el sistema.

    # Seleccionar voz según el idioma y el género
    if language == "en":
        # Buscar una voz en inglés
        for voice in voices:
            if "english" in voice.name.lower():  # Busca una voz que contenga "english" en su nombre
                engine.setProperty('voice', voices[2].id)
                break
    else:
        # Usar voces en español
        if gender == "male":
            # Selecciona una voz masculina en español
            # Nota: El índice de la voz (en este caso, 4) puede variar según el sistema.
            engine.setProperty('voice', voices[4].id)
        else:
            # Selecciona una voz femenina en español (por defecto)
            engine.setProperty('voice', voices[0].id)

    engine.say(text)  # Añade el texto a la cola de reproducción del motor.
    engine.runAndWait()  # Reproduce el texto en voz y espera a que termine.