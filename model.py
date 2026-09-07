from keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Input, GlobalAveragePooling1D, Add
from keras.models import Model
from keras.regularizers import l2
from constants import LENGTH_KEYPOINTS, MAX_LENGTH_FRAMES

NUM_EPOCH = 110

def transformer_encoder(inputs, d_model=128, num_heads=8, ff_dim=256, dropout=0.15):
    # Atención multi-cabeza
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    
    # Feed-forward
    ffn_output = Dense(ff_dim, activation="gelu")(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    
    return out2

def get_model(output_length: int):
    inputs = Input(shape=(MAX_LENGTH_FRAMES, LENGTH_KEYPOINTS))
    
    # Proyección inicial
    x = Dense(128)(inputs)
    
    # 4 bloques de encoder
    for _ in range(4):
        x = transformer_encoder(x)
    
    # Pooling y clasificación
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.15)(x)
    outputs = Dense(output_length, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
