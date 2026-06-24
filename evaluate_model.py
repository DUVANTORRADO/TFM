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

IMG_PATH = 'C:/Users/usuario/Desktop/VERSION 1.3 FINAL/modelo_lstm_lsc-main/img/'

def show_notification(message, duration=2000):
    notification = tk.Toplevel()
    notification.title("Notificación")
    notification.geometry("300x100")
    label = tk.Label(notification, text=message, font=("Arial", 12))
    label.pack(pady=20)
    notification.after(duration, notification.destroy)

def show_opencv_notification(frame, message, duration=2):
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
    global voice_gender
    voice_gender = gender if gender else "male" if voice_gender == "female" else "female"
    voice_gender_text = "hombre" if voice_gender == "male" else "mujer"
    show_notification(f"Voz cambiada a: {voice_gender_text}")
    if frame is not None:
        show_opencv_notification(frame, f"Voz cambiada a: {voice_gender_text}")

def save_translation(text):
    with open("historial_traducciones.txt", "a", encoding="utf-8") as file:
        file.write(text + "\n")

def overlay_image(background, overlay, x, y):
    h, w, _ = overlay.shape
    overlay_rgb = overlay[:, :, :3]
    mask = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (1 - mask) * background[y:y+h, x:x+w, c] + mask * overlay_rgb[:, :, c]

delete_last_img = cv2.imread(os.path.join(IMG_PATH, 'boton-eliminar.png'), cv2.IMREAD_UNCHANGED)
delete_last_img = cv2.resize(delete_last_img, (DELETE_LAST_X2 - DELETE_LAST_X1, DELETE_LAST_Y2 - DELETE_LAST_Y1))

delete_all_img = cv2.imread(os.path.join(IMG_PATH, 'boton-limpiar.png'), cv2.IMREAD_UNCHANGED)
delete_all_img = cv2.resize(delete_all_img, (DELETE_ALL_X2 - DELETE_ALL_X1, DELETE_ALL_Y2 - DELETE_ALL_Y1))

speak_img = cv2.imread(os.path.join(IMG_PATH, 'altavoz.png'), cv2.IMREAD_UNCHANGED)
speak_img = cv2.resize(speak_img, (SPEAK_X2 - SPEAK_X1, SPEAK_Y2 - SPEAK_Y1))

# Cargar la imagen del botón de traducción
translate_img = cv2.imread(os.path.join(IMG_PATH, 'traducir.png'), cv2.IMREAD_UNCHANGED)
translate_img = cv2.resize(translate_img, (TRANSLATE_X2 - TRANSLATE_X1, TRANSLATE_Y2 - TRANSLATE_Y1))

male_icon = cv2.imread(os.path.join(IMG_PATH, 'hombre.png'), cv2.IMREAD_UNCHANGED)
male_icon = cv2.resize(male_icon, (50, 40))

female_icon = cv2.imread(os.path.join(IMG_PATH, 'mujer.png'), cv2.IMREAD_UNCHANGED)
female_icon = cv2.resize(female_icon, (50, 40))

# Variables globales para almacenar el texto original y el estado de la traducción
original_lines = []  # Almacena el texto original
is_translated = False  # Indica si el texto está traducido o no
show_translate_button = False  # Controla la visibilidad del botón de traducción

def mouse_callback(event, x, y, flags, param):
    global lines, hovering, voice_gender, frame
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
            if TRANSLATE_X1 <= x <= TRANSLATE_X2 and TRANSLATE_Y1 <= y <= TRANSLATE_Y2:  # Nuevo botón de traducción
                translate_to_english()
        elif event == cv2.EVENT_MOUSEMOVE:
            hovering = next((k for k, (x1, y1, x2, y2) in {
                "delete_last": (DELETE_LAST_X1, DELETE_LAST_Y1, DELETE_LAST_X2, DELETE_LAST_Y2),
                "delete_all": (DELETE_ALL_X1, DELETE_ALL_Y1, DELETE_ALL_X2, DELETE_ALL_Y2),
                "speak": (SPEAK_X1, SPEAK_Y1, SPEAK_X2, SPEAK_Y2),
                "male_voice": (MALE_VOICE_BUTTON_X1, MALE_VOICE_BUTTON_Y1, MALE_VOICE_BUTTON_X2, MALE_VOICE_BUTTON_Y2),
                "female_voice": (FEMALE_VOICE_BUTTON_X1, FEMALE_VOICE_BUTTON_Y1, FEMALE_VOICE_BUTTON_X2, FEMALE_VOICE_BUTTON_Y2),
                "translate": (TRANSLATE_X1, TRANSLATE_Y1, TRANSLATE_X2, TRANSLATE_Y2)  # Nuevo botón de traducción
            }.items() if x1 <= x <= x2 and y1 <= y <= y2), None)
    except Exception as e:
        print(f"Error en mouse_callback: {e}")

def draw_tooltip(image, text, x, y, offset_y=0):
    if text:
        (w, h), _ = cv2.getTextSize(text, FONT, FONT_SIZE, 1)
        cv2.rectangle(image, (x, y - 25 + offset_y), (x + w + 10, y - 5 + offset_y), (50, 50, 50), -1)
        cv2.putText(image, text, (x + 5, y - 10 + offset_y), FONT, FONT_SIZE, (255, 255, 255), 1, cv2.LINE_AA)

def add_word(word, origin):
    global lines, is_translated, show_translate_button

    # Verificar si el texto está traducido
    if is_translated:
        messagebox.showwarning("Advertencia", "Debes revertir la traducción al español antes de agregar más palabras.")
        return  # Salir de la función sin agregar la palabra

    # Agregar la palabra si el texto no está traducido
    if lines and lines[-1] and lines[-1][-1][1] != origin:
        if not lines or len(lines[-1]) >= WORDS_PER_LINE:
            lines.append([])
        lines[-1].append(("-", origin))
    if not lines or len(lines[-1]) >= WORDS_PER_LINE:
        lines.append([])
    lines[-1].append((word, origin))
    show_translate_button = True  # Mostrar el botón de traducción
    update_translation_window(text_widget, lines)

def update_translation_window(text_widget, lines):
    text_widget.config(state=tk.NORMAL)
    text_widget.delete(1.0, tk.END)
    full_text = '\n'.join([' '.join([word for word, _ in line]) for line in lines])
    text_widget.insert(tk.END, full_text)
    text_widget.config(state=tk.DISABLED)
    text_widget.yview(tk.END)

def delete_last_word():
    global lines, show_translate_button, is_translated

    # Verificar si el texto está traducido al inglés
    if is_translated:
        messagebox.showwarning("Advertencia", "No se pueden eliminar palabras cuando el texto está traducido al inglés.")
        return  # Salir de la función sin eliminar palabras

    if lines:
        if lines[-1]:
            lines[-1].pop()
            if lines[-1] and lines[-1][-1][0] == "-":
                lines[-1].pop()
        if not lines[-1]:
            lines.pop()
    # Ocultar el botón solo si no hay palabras
    if not lines:
        show_translate_button = False
    update_translation_window(text_widget, lines)

def delete_all_words():
    global lines, show_translate_button, is_translated

    # Verificar si el texto está traducido al inglés
    if is_translated:
        messagebox.showwarning("Advertencia", "No se pueden eliminar palabras cuando el texto está traducido al inglés.")
        return  # Salir de la función sin eliminar palabras

    lines.clear()
    # Ocultar el botón solo si no hay palabras
    if not lines:
        show_translate_button = False
    update_translation_window(text_widget, lines)

def speak_text():
    global lines
    last_dash_index = -1  # Índice para rastrear la posición del último guion
    # Buscar el último guion en las líneas
    for i in range(len(lines) - 1, -1, -1):  # Recorrer desde el final hacia el inicio
        for j in range(len(lines[i]) - 1, -1, -1):  # Recorrer cada palabra en la línea
            if lines[i][j][0] == "-":  # Si encontramos un guion
                last_dash_index = (i, j)  # Guardar la posición del último guion
                break  # Salir del bucle interno
        if last_dash_index != -1:  # Si ya encontramos el último guion, salir del bucle externo
            break

    words_to_speak = []  # Lista para almacenar las palabras a reproducir
    if last_dash_index != -1:  # Si se encontró un guion
        i, j = last_dash_index  # Obtener la posición del último guion
        # Recorrer las líneas desde la posición del guion hasta el final
        for line in lines[i:]:
            for word, _ in line[j + 1:]:  # Tomar solo las palabras después del guion
                words_to_speak.append(word)
            j = -1  # Reiniciar el índice para las siguientes líneas
    else:  # Si no hay guion, reproducir todo el texto
        for line in lines:
            for word, _ in line:
                words_to_speak.append(word)

    if words_to_speak:  # Si hay palabras para reproducir
        text_to_speech(" ".join(words_to_speak), voice_gender)  # Reproducir la frase

def translate_to_english():
    global lines, original_lines, is_translated, show_translate_button
    translator = Translator()

    if not is_translated:
        # Guardar el texto original antes de traducir
        original_lines = lines.copy()
        # Traducir el texto al inglés
        full_text = '\n'.join([' '.join([word for word, _ in line]) for line in lines])
        translated_text = translator.translate(full_text, dest='en').text
        lines = [[(word, "traducción") for word in translated_text.split()]]
        is_translated = True  # Marcar como traducido
        show_translate_button = True  # Mostrar el botón de traducción
    else:
        # Restaurar el texto original
        lines = original_lines.copy()
        is_translated = False  # Marcar como no traducido
        # No cambiar show_translate_button aquí, ya que debe permanecer visible si hay palabras
        if not lines:  # Solo ocultar si no hay palabras
            show_translate_button = False

    # Actualizar la ventana de traducción
    update_translation_window(text_widget, lines)

def voice_to_text():
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

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

tooltip_active = True
face_previously_detected = False

def run_detection(model, text_widget, threshold=0.7):
    global lines, voice_gender, hovering, tooltip_active, face_previously_detected, frame, show_translate_button
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

                # Mostrar u ocultar el botón de traducción según el estado de show_translate_button
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

    voice_frame = tk.LabelFrame(translation_window, text="Cambiar Voz", padx=10, pady=10, bg="#2E2E2E", fg="white")
    voice_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    male_voice_button = tk.Button(voice_frame, text="Voz Masculina", command=lambda: change_voice(gender="male"), bg="#424242", fg="white")
    male_voice_button.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

    female_voice_button = tk.Button(voice_frame, text="Voz Femenina", command=lambda: change_voice(gender="female"), bg="#424242", fg="white")
    female_voice_button.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

    keyboard_frame = tk.LabelFrame(translation_window, text="Escribir Texto", padx=10, pady=10, bg="#2E2E2E", fg="white")
    keyboard_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    keyboard_label = tk.Label(keyboard_frame, text="Escribir texto:", bg="#2E2E2E", fg="white")
    keyboard_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

    keyboard_entry = tk.Entry(keyboard_frame, width=30, bg="#424242", fg="white")
    keyboard_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

    add_keyboard_button = tk.Button(keyboard_frame, text="Agregar texto", command=lambda: add_word(keyboard_entry.get(), "teclado"), bg="#424242", fg="white")
    add_keyboard_button.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

    video_frame = tk.LabelFrame(translation_window, text="Mostrar Seña", padx=10, pady=10, bg="#2E2E2E", fg="white")
    video_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    video_label = tk.Label(video_frame, text="Texto a seña:", bg="#2E2E2E", fg="white")
    video_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

    video_entry = tk.Entry(video_frame, width=30, bg="#424242", fg="white")
    video_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

    show_video_button = tk.Button(video_frame, text="Mostrar seña", command=lambda: show_sign_video(video_entry.get()), bg="#424242", fg="white")
    show_video_button.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

    text_widget = scrolledtext.ScrolledText(translation_window, wrap=tk.WORD, state=tk.DISABLED, bg="#424242", fg="white")
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    model_path = os.path.join(MODELS_PATH, MODEL_NAME)
    lstm_model = load_model(model_path)
    
    detection_thread = threading.Thread(target=run_detection, args=(lstm_model, text_widget))
    detection_thread.daemon = True
    detection_thread.start()
    
    translation_window.mainloop()