from keras.layers import Dense, Dropout, Input, LSTM, Bidirectional, GlobalAveragePooling1D
from keras.models import Model
from keras.regularizers import l2
from constants import LENGTH_KEYPOINTS, MAX_LENGTH_FRAMES

NUM_EPOCH = 110

def get_model(output_length: int):
    inputs = Input(shape=(MAX_LENGTH_FRAMES, LENGTH_KEYPOINTS))
    
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(inputs)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(output_length, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model