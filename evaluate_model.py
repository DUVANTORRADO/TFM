import os
import tkinter as tk
from tkinter import messagebox, scrolledtext
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from helpers import draw_keypoints, extract_keypoints, get_actions, mediapipe_detection, save_txt, there_hand
from text_to_speech import text_to_speech
from constants import *
import threading
import speech_recognition as sr
import time
from googletrans import Translator  # Importar la librería para traducción

# Ruta de la carpeta que contiene las imágenes de los botones
IMG_PATH = 'C:/Users/usuario/Desktop/VERSION 1.3 FINAL/modelo_lstm_lsc-main/img/'

def show_notification(message, duration=2000):
    """
    Muestra una notificación en una ventana emergente de Tkinter.
    
    Args:
        message (str): El mensaje a mostrar.
        duration (int): Duración en milisegundos antes de que la ventana se cierre automáticamente.
    """
    notification = tk.Toplevel()
    notification.title("Notificación")
    notification.geometry("300x100")
    label = tk.Label(notification, text=message, font=("Arial", 12))
    label.pack(pady=20)
    notification.after(duration, notification.destroy)

def show_opencv_notification(frame, message, duration=2):
    """
    Muestra una notificación en una ventana de OpenCV.
    
    Args:
        frame: El frame de la imagen donde se mostrará la notificación.
        message (str): El mensaje a mostrar.
        duration (int): Duración en segundos antes de que la notificación desaparezca.
    """
    height, width = frame.shape[:2]
    text_size = cv2.getTextSize(message, FONT, FONT_SIZE, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(frame, message, (text_x, text_y), FONT, FONT_SIZE, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Traductor de Senas Colombiano', frame)
    start_time = time.time()
    while time.time() - start_time < duration:
        cv2.waitKey(10)
    cv2.putText(frame, "", (text_x, text_y), FONT, FONT_SIZE, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Traductor de Senas Colombiano', frame)

def change_voice(frame=None, gender=None):
    """
    Cambia el género de la voz utilizada para la conversión de texto a voz.
    
    Args:
        frame: El frame de la imagen donde se mostrará la notificación (opcional).
        gender (str): El género de la voz ("male" o "female").
    """
    global voice_gender
    voice_gender = gender if gender else "male" if voice_gender == "female" else "female"
    voice_gender_text = "hombre" if voice_gender == "male" else "mujer"
    show_notification(f"Voz cambiada a: {voice_gender_text}")
    if frame is not None:
        show_opencv_notification(frame, f"Voz cambiada a: {voice_gender_text}")

def save_translation(text):
    """
    Guarda el texto traducido en un archivo de historial.
    
    Args:
        text (str): El texto a guardar.
    """
    with open("historial_traducciones.txt", "a", encoding="utf-8") as file:
        file.write(text + "\n")

def overlay_image(background, overlay, x, y):
    """
    Superpone una imagen sobre otra en una posición específica.
    
    Args:
        background: La imagen de fondo.
        overlay: La imagen a superponer.
        x (int): Coordenada X donde se colocará la imagen superpuesta.
        y (int): Coordenada Y donde se colocará la imagen superpuesta.
    """
    h, w, _ = overlay.shape
    overlay_rgb = overlay[:, :, :3]
    mask = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (1 - mask) * background[y:y+h, x:x+w, c] + mask * overlay_rgb[:, :, c]

# Cargar imágenes de los botones
delete_last_img = cv2.imread(os.path.join(IMG_PATH, 'boton-eliminar.png'), cv2.IMREAD_UNCHANGED)
delete_last_img = cv2.resize(delete_last_img, (DELETE_LAST_X2 - DELETE_LAST_X1, DELETE_LAST_Y2 - DELETE_LAST_Y1))

delete_all_img = cv2.imread(os.path.join(IMG_PATH, 'boton-limpiar.png'), cv2.IMREAD_UNCHANGED)
delete_all_img = cv2.resize(delete_all_img, (DELETE_ALL_X2 - DELETE_ALL_X1, DELETE_ALL_Y2 - DELETE_ALL_Y1))

speak_img = cv2.imread(os.path.join(IMG_PATH, 'altavoz.png'), cv2.IMREAD_UNCHANGED)
speak_img = cv2.resize(speak_img, (SPEAK_X2 - SPEAK_X1, SPEAK_Y2 - SPEAK_Y1))

translate_img = cv2.imread(os.path.join(IMG_PATH, 'traducir.png'), cv2.IMREAD_UNCHANGED)
translate_img = cv2.resize(translate_img, (TRANSLATE_X2 - TRANSLATE_X1, TRANSLATE_Y2 - TRANSLATE_Y1))

male_icon = cv2.imread(os.path.join(IMG_PATH, 'hombre.png'), cv2.IMREAD_UNCHANGED)
male_icon = cv2.resize(male_icon, (50, 40))

female_icon = cv2.imread(os.path.join(IMG_PATH, 'mujer.png'), cv2.IMREAD_UNCHANGED)
female_icon = cv2.resize(female_icon, (50, 40))

# Variables globales para almacenar el texto original y el estado de traducción
original_lines = []  # Almacena el texto original
is_translated = False  # Indica si el texto está traducido o no
show_translate_button = False  # Controla la visibilidad del botón de traducción

def detect_language(text):
    """
    Detecta el idioma del texto utilizando la librería googletrans.
    
    Args:
        text (str): El texto a analizar.
    
    Returns:
        str: El código del idioma detectado.
    """
    translator = Translator()
    try:
        detected_lang = translator.detect(text).lang
        return detected_lang
    except Exception as e:
        print(f"Error detectando idioma: {e}")
        return "es"  # Por defecto, asumimos español

def translate_text(text, src_lang, dest_lang):
    """
    Traduce el texto de un idioma a otro utilizando la librería googletrans.
    
    Args:
        text (str): El texto a traducir.
        src_lang (str): El idioma de origen.
        dest_lang (str): El idioma de destino.
    
    Returns:
        str: El texto traducido.
    """
    translator = Translator()
    try:
        translated_text = translator.translate(text, src=src_lang, dest=dest_lang).text
        return translated_text
    except Exception as e:
        print(f"Error traduciendo texto: {e}")
        return text  # Si falla, devuelve el texto original

def add_word(word, origin):
    """
    Agrega una palabra o frase al texto traducido.
    
    Args:
        word (str): La palabra o frase a agregar.
        origin (str): El origen de la palabra ("teclado", "voz", "señas").
    """
    global lines, is_translated, show_translate_button

    # Detectar el idioma de la palabra
    src_lang = detect_language(word)
    if src_lang == "en":  # Si la palabra está en inglés, traducirla al español
        word = translate_text(word, src_lang, "es")

    # Dividir la frase en palabras individuales
    words = word.split()
    for w in words:
        if lines and lines[-1] and lines[-1][-1][1] != origin:
            if not lines or len(lines[-1]) >= WORDS_PER_LINE:
                lines.append([])
            lines[-1].append(("-", origin))
        if not lines or len(lines[-1]) >= WORDS_PER_LINE:
            lines.append([])
        lines[-1].append((w, origin))  # Agregar cada palabra individualmente

    show_translate_button = True
    update_translation_window(text_widget, lines)

    # Limpiar la casilla de entrada después de agregar la palabra
    if origin == "teclado":
        keyboard_entry.delete(0, tk.END)  # Borrar el contenido del Entry

def toggle_translation():
    """
    Alterna entre la traducción del texto y su reversión al texto original.
    """
    global lines, original_lines, is_translated, show_translate_button, translate_button

    if not lines:
        return  # No hay texto para traducir

    # Obtener el texto completo
    full_text = '\n'.join([' '.join([word for word, _ in line]) for line in lines])

    # Buscar el último guion en el texto
    last_dash_index = full_text.rfind('-')
    if last_dash_index == -1:
        # Si no hay guion, traducir todo el texto
        text_to_translate = full_text
    else:
        # Extraer el texto después del último guion
        text_to_translate = full_text[last_dash_index + 1:].strip()

    # Detectar el idioma del texto a traducir
    src_lang = detect_language(text_to_translate)
    dest_lang = "en" if src_lang == "es" else "es"  # Traducir al idioma opuesto

    if is_translated:
        # Si ya está traducido, revertir al texto original
        lines = original_lines.copy()
    else:
        # Guardar el texto original antes de traducir
        original_lines = lines.copy()
        # Traducir solo la última frase
        translated_text = translate_text(text_to_translate, src_lang, dest_lang)
        # Reemplazar la última frase con la traducción
        if last_dash_index == -1:
            # Si no hay guion, reemplazar todo el texto
            lines = [[(word, "traducción") for word in translated_text.split()]]
        else:
            # Dividir el texto en partes antes y después del último guion
            before_last_dash = full_text[:last_dash_index + 1].strip()
            after_last_dash = translated_text
            # Actualizar las líneas con la traducción
            lines = []
            # Agregar la parte antes del guion
            for line in before_last_dash.split('\n'):
                lines.append([(word, "original") for word in line.split()])
            # Agregar la traducción de la última frase
            lines.append([(word, "traducción") for word in after_last_dash.split()])

    is_translated = not is_translated  # Cambiar el estado de traducción

    # Cambiar el color del botón según el estado de traducción
    if is_translated:
        translate_button.config(bg="green")  # Cambiar a verde cuando la traducción está activa
    else:
        translate_button.config(bg="#424242")  # Volver al color original cuando no está activa

    # Actualizar la ventana de traducción
    update_translation_window(text_widget, lines)

def mouse_callback(event, x, y, flags, param):
    """
    Maneja los eventos del mouse en la ventana de OpenCV.
    
    Args:
        event: El tipo de evento del mouse.
        x (int): Coordenada X del evento.
        y (int): Coordenada Y del evento.
        flags: Flags adicionales del evento.
        param: Parámetros adicionales del evento.
    """
    global lines, hovering, voice_gender, frame, is_translated

    try:
        if event == cv2.EVENT_LBUTTONDOWN:
            if DELETE_LAST_X1 <= x <= DELETE_LAST_X2 and DELETE_LAST_Y1 <= y <= DELETE_LAST_Y2:
                delete_last_word()
            if DELETE_ALL_X1 <= x <= DELETE_ALL_X2 and DELETE_ALL_Y1 <= y <= DELETE_ALL_Y2:
                delete_all_words()
            if SPEAK_X1 <= x <= SPEAK_X2 and SPEAK_Y1 <= y <= SPEAK_Y2:
                speak_text()
            if MALE_VOICE_BUTTON_X1 <= x <= MALE_VOICE_BUTTON_X2 and MALE_VOICE_BUTTON_Y1 <= y <= MALE_VOICE_BUTTON_Y2:
                change_voice(frame=frame, gender="male")
            if FEMALE_VOICE_BUTTON_X1 <= x <= FEMALE_VOICE_BUTTON_X2 and FEMALE_VOICE_BUTTON_Y1 <= y <= FEMALE_VOICE_BUTTON_Y2:
                change_voice(frame=frame, gender="female")
            if TRANSLATE_X1 <= x <= TRANSLATE_X2 and TRANSLATE_Y1 <= y <= TRANSLATE_Y2:
                toggle_translation()  # Cambiar entre traducción y reversión
        elif event == cv2.EVENT_MOUSEMOVE:
            hovering = next((k for k, (x1, y1, x2, y2) in {
                "delete_last": (DELETE_LAST_X1, DELETE_LAST_Y1, DELETE_LAST_X2, DELETE_LAST_Y2),
                "delete_all": (DELETE_ALL_X1, DELETE_ALL_Y1, DELETE_ALL_X2, DELETE_ALL_Y2),
                "speak": (SPEAK_X1, SPEAK_Y1, SPEAK_X2, SPEAK_Y2),
                "male_voice": (MALE_VOICE_BUTTON_X1, MALE_VOICE_BUTTON_Y1, MALE_VOICE_BUTTON_X2, MALE_VOICE_BUTTON_Y2),
                "female_voice": (FEMALE_VOICE_BUTTON_X1, FEMALE_VOICE_BUTTON_Y1, FEMALE_VOICE_BUTTON_X2, FEMALE_VOICE_BUTTON_Y2),
                "translate": (TRANSLATE_X1, TRANSLATE_Y1, TRANSLATE_X2, TRANSLATE_Y2)
            }.items() if x1 <= x <= x2 and y1 <= y <= y2), None)
    except Exception as e:
        print(f"Error en mouse_callback: {e}")

def draw_tooltip(image, text, x, y, offset_y=0):
    """
    Dibuja un tooltip en la imagen.
    
    Args:
        image: La imagen donde se dibujará el tooltip.
        text (str): El texto del tooltip.
        x (int): Coordenada X del tooltip.
        y (int): Coordenada Y del tooltip.
        offset_y (int): Desplazamiento vertical del tooltip.
    """
    if text:
        (w, h), _ = cv2.getTextSize(text, FONT, FONT_SIZE, 1)
        cv2.rectangle(image, (x, y - 25 + offset_y), (x + w + 10, y - 5 + offset_y), (50, 50, 50), -1)
        cv2.putText(image, text, (x + 5, y - 10 + offset_y), FONT, FONT_SIZE, (255, 255, 255), 1, cv2.LINE_AA)

def update_translation_window(text_widget, lines):
    """
    Actualiza la ventana de traducción con el texto proporcionado.
    
    Args:
        text_widget: El widget de texto de Tkinter.
        lines: Las líneas de texto a mostrar.
    """
    text_widget.config(state=tk.NORMAL)
    text_widget.delete(1.0, tk.END)
    full_text = '\n'.join([' '.join([word for word, _ in line]) for line in lines])
    text_widget.insert(tk.END, full_text)
    text_widget.config(state=tk.DISABLED)
    text_widget.yview(tk.END)

def delete_last_word():
    """
    Elimina la última palabra del texto traducido.
    """
    global lines, show_translate_button, is_translated

    if is_translated:
        messagebox.showwarning("Advertencia", "No se pueden eliminar palabras cuando el texto está traducido.")
        return

    if lines:
        if lines[-1]:
            lines[-1].pop()  # Eliminar la última palabra
            if lines[-1] and lines[-1][-1][0] == "-":
                lines[-1].pop()  # Eliminar el guion si está presente
        if not lines[-1]:
            lines.pop()  # Eliminar la línea si está vacía
    if not lines:
        show_translate_button = False
    update_translation_window(text_widget, lines)

def delete_all_words():
    """
    Elimina todas las palabras del texto traducido.
    """
    global lines, show_translate_button, is_translated

    if is_translated:
        messagebox.showwarning("Advertencia", "No se pueden eliminar palabras cuando el texto está traducido.")
        return

    lines.clear()
    if not lines:
        show_translate_button = False
    update_translation_window(text_widget, lines)

def speak_text():
    """
    Reproduce el texto traducido utilizando la conversión de texto a voz.
    """
    global lines, is_translated, voice_gender

    language = "en" if is_translated else "es"

    last_dash_index = -1
    for i in range(len(lines) - 1, -1, -1):
        for j in range(len(lines[i]) - 1, -1, -1):
            if lines[i][j][0] == "-":
                last_dash_index = (i, j)
                break
        if last_dash_index != -1:
            break

    words_to_speak = []
    if last_dash_index != -1:
        i, j = last_dash_index
        for line in lines[i:]:
            for word, _ in line[j + 1:]:
                words_to_speak.append(word)
            j = -1
    else:
        for line in lines:
            for word, _ in line:
                words_to_speak.append(word)

    if words_to_speak:
        text_to_speech(" ".join(words_to_speak), voice_gender, language)

def voice_to_text():
    """
    Convierte la voz en texto utilizando el reconocimiento de voz.
    """
    listening_window = tk.Toplevel()
    listening_window.title("Escuchando")
    listening_window.geometry("200x100")
    listening_window.configure(bg="#2E2E2E")
    listening_label = tk.Label(listening_window, text="Escuchando...", font=("Arial", 12), bg="#2E2E2E", fg="white")
    listening_label.pack(pady=20)
    listening_window.update()
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio, language="es-ES")
            add_word(text, "voz")
        except sr.UnknownValueError:
            print("No se pudo entender el audio")
        except sr.RequestError as e:
            print(f"Error al solicitar resultados del servicio de reconocimiento de voz: {e}")
        except sr.WaitTimeoutError:
            print("Tiempo de espera agotado")
        finally:
            listening_window.destroy()

def show_sign_video(word):
    """
    Muestra un video de la seña correspondiente a una palabra.
    
    Args:
        word (str): La palabra para la cual se mostrará la seña.
    """
    video_path = f"videos/{word}.mp4"
    if not os.path.exists(video_path):
        messagebox.showerror("Error", f"No se encontró un video para la palabra '{word}'.")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "No se pudo abrir el video.")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f"Seña para: {word}", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()

# Inicialización de MediaPipe para la detección de rostros
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

tooltip_active = True
face_previously_detected = False

def run_detection(model, text_widget, threshold=0.7):
    """
    Ejecuta la detección de señas en tiempo real utilizando el modelo LSTM.
    
    Args:
        model: El modelo LSTM cargado.
        text_widget: El widget de texto de Tkinter para mostrar las traducciones.
        threshold (float): Umbral de confianza para la detección de señas.
    """
    global lines, voice_gender, hovering, tooltip_active, face_previously_detected, frame, show_translate_button, is_translated
    actions = get_actions(DATA_PATH)
    kp_sequence = []
    count_frame = 0
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            raise RuntimeError("No se pudo acceder a la cámara.")

        cv2.namedWindow('Traductor de Senas Colombiano')
        cv2.setMouseCallback('Traductor de Senas Colombiano', mouse_callback)

        while video.isOpened():
            try:
                _, frame = video.read()
                height, width = frame.shape[:2]
                new_width = int(width * 1.2)
                new_height = int(height * 1.2)
                frame = cv2.resize(frame, (new_width, new_height))

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_results = face_detection.process(rgb_frame)
                has_face = face_results.detections is not None

                image, results = mediapipe_detection(frame, holistic_model)
                draw_keypoints(image, results)
                
                if there_hand(results):
                    tooltip_active = False
                    kp_sequence.append(extract_keypoints(results))
                    if len(kp_sequence) > MAX_LENGTH_FRAMES:
                        count_frame += 1
                else:
                    if count_frame >= MIN_LENGTH_FRAMES:
                        res = model.predict(np.expand_dims(kp_sequence[-MAX_LENGTH_FRAMES:], axis=0))[0]
                        if res[np.argmax(res)] > threshold:
                            sent = actions[np.argmax(res)]
                            print(f"Palabra detectada: {sent}")
                            text_to_speech(sent, voice_gender)
                            save_translation(sent)
                            add_word(sent, "señas")
                        count_frame = 0
                        kp_sequence = []
                
                cv2.rectangle(image, (0, 460), (770, 575), (80, 10, 40), -1)

                y0, dy = 480, 25
                for line in lines[-MAX_LINES:]:
                    x0 = 10
                    for word, origin in line:
                        color = (255, 255, 255) if origin == "señas" else (0, 255, 0)
                        cv2.putText(image, word, (x0, y0), FONT, FONT_SIZE, color, 2, cv2.LINE_AA)
                        (w, _), _ = cv2.getTextSize(word, FONT, FONT_SIZE, 2)
                        x0 += w + 10
                    y0 += dy

                overlay_image(image, delete_last_img, DELETE_LAST_X1, DELETE_LAST_Y1)
                overlay_image(image, delete_all_img, DELETE_ALL_X1, DELETE_ALL_Y1)
                overlay_image(image, speak_img, SPEAK_X1, SPEAK_Y1)

                if show_translate_button:
                    overlay_image(image, translate_img, TRANSLATE_X1, TRANSLATE_Y1)

                overlay_image(image, male_icon, MALE_VOICE_BUTTON_X1, MALE_VOICE_BUTTON_Y1)
                overlay_image(image, female_icon, FEMALE_VOICE_BUTTON_X1, FEMALE_VOICE_BUTTON_Y1)

                if not face_previously_detected and has_face:
                    tooltip_active = True
                face_previously_detected = has_face

                if tooltip_active and has_face:
                    draw_tooltip(image, "Usa tus manos para comunicarte con lengua de señas", 10, 45)

                if hovering:
                    draw_tooltip(image, tooltips[hovering], 10, 45, offset_y=25)
                
                cv2.imshow('Traductor de Senas Colombiano', image)
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
            except Exception as e:
                print(f"Error en el bucle principal: {e}")
                
        video.release()
        cv2.destroyAllWindows()

# ... (código anterior sin cambios)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    response = messagebox.askokcancel("Aviso", "Se desplegará la cámara del dispositivo donde podrás empezar a traducir lenguaje de señas a texto y voz. ¿Desea continuar?")

    if not response:
        exit()
    
    translation_window = tk.Toplevel()
    translation_window.title("Traducciones en Tiempo Real")
    translation_window.geometry("600x500")
    translation_window.configure(bg="#2E2E2E")

    # Frame de controles
    control_frame = tk.LabelFrame(translation_window, text="Controles", padx=10, pady=10, bg="#2E2E2E", fg="white")
    control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    delete_last_button = tk.Button(control_frame, text="Eliminar última palabra", command=delete_last_word, bg="#424242", fg="white")
    delete_last_button.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

    delete_all_button = tk.Button(control_frame, text="Eliminar todas las palabras", command=delete_all_words, bg="#424242", fg="white")
    delete_all_button.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

    speak_button = tk.Button(control_frame, text="Reproducir texto", command=speak_text, bg="#424242", fg="white")
    speak_button.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

    voice_to_text_button = tk.Button(control_frame, text="Voz a Texto", command=voice_to_text, bg="#424242", fg="white")
    voice_to_text_button.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

    # Frame para cambiar la voz
    voice_frame = tk.LabelFrame(translation_window, text="Cambiar Voz", padx=10, pady=10, bg="#2E2E2E", fg="white")
    voice_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    male_voice_button = tk.Button(voice_frame, text="Voz Masculina", command=lambda: change_voice(gender="male"), bg="#424242", fg="white")
    male_voice_button.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

    female_voice_button = tk.Button(voice_frame, text="Voz Femenina", command=lambda: change_voice(gender="female"), bg="#424242", fg="white")
    female_voice_button.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

    # Frame para escribir texto
    keyboard_frame = tk.LabelFrame(translation_window, text="Escribir Texto", padx=10, pady=10, bg="#2E2E2E", fg="white")
    keyboard_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    keyboard_label = tk.Label(keyboard_frame, text="Escribir texto:", bg="#2E2E2E", fg="white")
    keyboard_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

    keyboard_entry = tk.Entry(keyboard_frame, width=30, bg="#424242", fg="white")
    keyboard_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

    add_keyboard_button = tk.Button(keyboard_frame, text="Agregar texto", command=lambda: add_word(keyboard_entry.get(), "teclado"), bg="#424242", fg="white")
    add_keyboard_button.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

    # Frame para mostrar la seña (texto a video)
    video_frame = tk.LabelFrame(translation_window, text="Mostrar Seña", padx=10, pady=10, bg="#2E2E2E", fg="white")
    video_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    video_label = tk.Label(video_frame, text="Texto a seña:", bg="#2E2E2E", fg="white")
    video_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

    video_entry = tk.Entry(video_frame, width=30, bg="#424242", fg="white")
    video_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

    show_video_button = tk.Button(video_frame, text="Mostrar seña", command=lambda: show_sign_video(video_entry.get()), bg="#424242", fg="white")
    show_video_button.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

    # Frame para el área de texto y el botón de traducción
    text_frame = tk.Frame(translation_window, bg="#2E2E2E")
    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Botón de traducción colocado arriba del text_widget
    translate_button = tk.Button(text_frame, text="Traducir a inglés", command=toggle_translation, bg="#424242", fg="white")
    translate_button.pack(side=tk.TOP, pady=5)  # Colocado arriba del text_widget

    # Área de texto (text_widget)
    text_widget = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, state=tk.DISABLED, bg="#424242", fg="white")
    text_widget.pack(fill=tk.BOTH, expand=True)

    # Cargar el modelo LSTM
    model_path = os.path.join(MODELS_PATH, MODEL_NAME)
    lstm_model = load_model(model_path)
    
    # Iniciar el hilo de detección
    detection_thread = threading.Thread(target=run_detection, args=(lstm_model, text_widget))
    detection_thread.daemon = True
    detection_thread.start()
    
    # Ejecutar la ventana principal
    translation_window.mainloop()