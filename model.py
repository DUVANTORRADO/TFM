
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import LSTM, Dense
from constants import LENGTH_KEYPOINTS, MAX_LENGTH_FRAMES

NUM_EPOCH = 110

def get_model(output_lenght: int):
    
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(MAX_LENGTH_FRAMES, LENGTH_KEYPOINTS), kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(output_lenght, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model