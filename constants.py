import os
import cv2

# PATHS
ROOT_PATH = os.getcwd()
FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "frame_actions")
DATA_PATH = os.path.join(ROOT_PATH, "data")
MODELS_PATH = os.path.join(ROOT_PATH, "models")

# MODEL PARAMETERS
MAX_LENGTH_FRAMES = 15
LENGTH_KEYPOINTS = 1662
MIN_LENGTH_FRAMES = 5
MODEL_NAME = f"actions_{MAX_LENGTH_FRAMES}.keras"

# SHOW IMAGE PARAMETERS
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

# OPENCV DISPLAY CONSTANTS
MAX_LINES = 4
WORDS_PER_LINE = 6  # Mostrar 5 palabras por renglón

# BUTTON POSITIONS
DELETE_LAST_X1, DELETE_LAST_Y1 = 705, 445
DELETE_LAST_X2, DELETE_LAST_Y2 = 765, 515 

DELETE_ALL_X1, DELETE_ALL_Y1 = 720, 520  
DELETE_ALL_X2, DELETE_ALL_Y2 = 765, 570

SPEAK_X1, SPEAK_Y1 = 6, 430 
SPEAK_X2, SPEAK_Y2 = 40, 460

# VOICE STATE
voice_gender = "female"

# TOOLTIPS
# Definir tooltips para los botones de voz
tooltips = {
    "delete_last": "Eliminar última palabra",
    "delete_all": "Eliminar todas las palabras",
    "speak": "Reproducir texto",
    "male_voice": "Cambiar a voz masculina",
    "female_voice": "Cambiar a voz femenina",
    "translate": "Traducir texto al inglés"
    
}

# HOVERING STATE
hovering = None

# LINES STORAGE
lines = []  # Almacena tuplas: (palabra, origen)

# CONSTANTES ADICIONALES
MALE_VOICE_BUTTON_X1 = 710
MALE_VOICE_BUTTON_Y1 = 415
MALE_VOICE_BUTTON_X2 = MALE_VOICE_BUTTON_X1 + 50  # Ancho del botón
MALE_VOICE_BUTTON_Y2 = MALE_VOICE_BUTTON_Y1 + 40  # Alto del botón

# Coordenadas del botón de traducción
TRANSLATE_X1, TRANSLATE_Y1 = 710, 10
TRANSLATE_X2, TRANSLATE_Y2 = 750, 50

FEMALE_VOICE_BUTTON_X1 = 660
FEMALE_VOICE_BUTTON_Y1 = 415
FEMALE_VOICE_BUTTON_X2 = FEMALE_VOICE_BUTTON_X1 + 50  # Ancho del botón
FEMALE_VOICE_BUTTON_Y2 = FEMALE_VOICE_BUTTON_Y1 + 40  # Alto del botón