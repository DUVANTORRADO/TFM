import pyttsx3  

def text_to_speech(text, gender="female"):

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')  # Obtiene la lista de voces disponibles en el sistema.

    # Seleccionar voz masculina o femenina
    if gender == "male":
        # Si el género es "male", selecciona una voz masculina.
        # Nota: El índice de la voz (en este caso, 4) puede variar según el sistema.
        engine.setProperty('voice', voices[4].id)
    else:
        # Si el género no es "male", selecciona una voz femenina (por defecto).
        engine.setProperty('voice', voices[0].id)

    engine.say(text)  # Añade el texto a la cola de reproducción del motor.
    engine.runAndWait()  # Reproduce el texto en voz y espera a que termine.