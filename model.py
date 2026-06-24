from keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Input, GlobalAveragePooling1D
from keras.models import Model
from keras.regularizers import l2
from constants import LENGTH_KEYPOINTS, MAX_LENGTH_FRAMES

NUM_EPOCH = 35 # Los transformers a veces necesitan un poco más de épocas

def get_model(output_length: int):
    # Definimos la entrada (15 frames, 1662 puntos clave)
    inputs = Input(shape=(MAX_LENGTH_FRAMES, LENGTH_KEYPOINTS))
    
    # --- MECANISMO DE ATENCIÓN (El "cerebro" del Transformer) ---
    # Esto permite que el modelo se enfoque en movimientos clave
    attention_out = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    attention_out = Dropout(0.5)(attention_out)
    
    # Sumamos la entrada original para no perder información (Conexión Residual)
    x = LayerNormalization(epsilon=1e-6)(attention_out + inputs)
    
    # --- CAPA DE PROCESAMIENTO EXTRA ---
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    
    # --- CLASIFICACIÓN FINAL ---
    # Reducimos los 15 frames a un solo vector para decidir qué palabra es
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    outputs = Dense(output_length, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model