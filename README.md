# Traductor de Lengua de Señas Colombiano (LSC) a Texto y Voz

#Implementación y Evaluación de Redes de Transformer Para el Reconocimiento de Lenguaje de Señas Colombiano Aplicado a la Base de Datos del INSOR

Este proyecto es parte del Trabajo Fin de Máster (TFM) en Inteligencia Artificial. El sistema implementa y evalúa arquitecturas de redes neuronales para la traducción de señas de la LSC, comparando el rendimiento entre modelos basados en **Transformers**, **CNN** y **LSTM**.

##  Estructura del Repositorio
Para facilitar la lectura y el mantenimiento, el proyecto está organizado de la siguiente manera:

- `src/`: Scripts principales de ejecución.
  - `capture_samples.py`: Captura y preprocesamiento de muestras.
  - `create_keypoints.py`: Generación de puntos clave (keypoints) con MediaPipe.
  - `training_model.py`: Entrenamiento del modelo.
  - `evaluate_model.py`: Evaluación del rendimiento del modelo.
- `models/`: Definición de arquitecturas (model.py, model_lstm.py) y parámetros.
- `data/`: Almacenamiento de datasets y archivos de keypoints (.h5).
- `outputs/`: Resultados, historial de traducciones y logs.
- `img/`: Recursos visuales del proyecto.

## Instalación y Reproducibilidad
Para garantizar la reproducibilidad fiel de los experimentos, hemos fijado las versiones exactas de todas las dependencias.

1. Clona este repositorio:
   ```bash
   git clone [https://github.com/DUVANTORRADO/TFM.git](https://github.com/DUVANTORRADO/TFM.git)
   cd TFM
   
Instala las librerías con las versiones especificadas:

```bash
pip install -r requirements.txt
```

 Flujo de Trabajo
Para ejecutar el experimento desde cero, sigue estos pasos:

1. Captura: python src/capture_samples.py

2. Keypoints: python src/create_keypoints.py

3. Entrenamiento: python src/training_model.py

4. Evaluación: python src/evaluate_model.py

 Autor
Luis Duvan Torrado Mora Universidad Francisco de Paula Santander Ocaña, Colombia
Correo: torradoduban.18@gmail.com

