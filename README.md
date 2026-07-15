Este es un modelo de una red neuronal que traduce Lengua de Señas Colombiano (LSC) a texto (y voz). Utilicé MediaPipe para obtener los puntos de la seña y para el entrenamiento usé TensorFlow y Keras.

## SCRIPTS PRINCIPALES
- capture_samples.py → captura las muestras y las ubica en la carpeta frame_actions.
- create_keypoints.py → crea los keypoints que se usarán en el entrenamiento.
- training_model.py → entrena la red neuronal.
- evaluate_model.py → donde se realiza la prueba de la red neuronal.

## SCRIPTS SECUNDARIOS
- model.py → aquí se ajusta el modelo de la red neuronal.
- constants.py → ajustes de la red neuronal.
- helpers.py → funciones que se utilizan en los scripts principales.

## Pasos para probar la red neuronal
1. Capturar las muestras con capture_samples.py
2. Generar los .h5 (keypoints) de cada palabra con create_keypoints.py
3. Entrenar el modelo con training_model.py
4. Realizar pruebas con evaluate_model.py

## instalacion
Agregar python a las variables de entorno editando la variable path por ejemplo

C:\Users\Administrador\AppData\Local\Programs\Python\Python39  ------> debe quedar de primero
C:\Users\Administrador\AppData\Local\Programs\Python\Python39\Scripts


instalar tensorflow 

pip install tensorflow==2.10.1

y luego las otras librerias

 pip install mediapipe==0.10.11, numpy==1.26.4, tables==3.9.2, opencv-python==4.9.0.80, pandas==2.2.1, protobuf==3.20.3, keras==2.10.0, h5py==3.10.0, flatbuffers==24.3.7, gTTS==2.5.1, pygame==2.5.2

crear primero todas las palabras para predecir
luego crear los keypoints
despues entrenar la red
por ultimo evaluar el modelo

## LUIS DUVAN TORRADO MORA - UNIVERSIDAD FRANSICO DE PAULA SANTANDER OCAÑA


Explicación General del Código
Este código es un sistema de traducción de lenguaje de señas a texto y voz en tiempo real. Utiliza una combinación de tecnologías como OpenCV para la captura de video, MediaPipe para la detección de puntos clave en las manos y el rostro, y un modelo LSTM para la clasificación de las señas. Además, cuenta con una interfaz gráfica desarrollada con Tkinter para mostrar las traducciones y permitir la interacción del usuario.

Funciones Principales
show_notification: Muestra una notificación emergente en la interfaz gráfica.

show_opencv_notification: Muestra una notificación en la ventana de OpenCV.

change_voice: Cambia el género de la voz utilizada para la conversión de texto a voz.

save_translation: Guarda el texto traducido en un archivo de historial.

overlay_image: Superpone una imagen sobre otra en una posición específica.

detect_language: Detecta el idioma del texto utilizando la librería googletrans.

translate_text: Traduce el texto de un idioma a otro utilizando la librería googletrans.

add_word: Agrega una palabra o frase al texto traducido.

toggle_translation: Alterna entre la traducción del texto y su reversión al texto original.

mouse_callback: Maneja los eventos del mouse en la ventana de OpenCV.

draw_tooltip: Dibuja un tooltip en la imagen.

update_translation_window: Actualiza la ventana de traducción con el texto proporcionado.

delete_last_word: Elimina la última palabra del texto traducido.

delete_all_words: Elimina todas las palabras del texto traducido.

speak_text: Reproduce el texto traducido utilizando la conversión de texto a voz.

voice_to_text: Convierte la voz en texto utilizando el reconocimiento de voz.

show_sign_video: Muestra un video de la seña correspondiente a una palabra.

run_detection: Ejecuta la detección de señas en tiempo real utilizando el modelo LSTM.

Interfaz Gráfica
La interfaz gráfica está construida con Tkinter y permite al usuario interactuar con el sistema de traducción. Incluye botones para eliminar palabras, reproducir texto, cambiar la voz, y más. También muestra el texto traducido en tiempo real y permite la traducción inversa.

Detección de Señas
La detección de señas se realiza mediante un modelo LSTM que ha sido entrenado previamente. El modelo recibe una secuencia de puntos clave detectados por MediaPipe y predice la seña correspondiente. Si la confianza de la predicción supera un umbral, la seña se traduce a texto y se reproduce en voz alta.

Conclusión
Este código es un sistema completo para la traducción de lenguaje de señas a texto y voz. Combina técnicas de visión por computadora, aprendizaje automático, y procesamiento de lenguaje natural para ofrecer una solución integral. La documentación proporcionada debería ayudar a otros programadores a entender y extender el código según sea necesario.