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