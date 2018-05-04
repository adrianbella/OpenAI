from keras import Sequential
from keras.layers import Convolution2D
from keras.layers import Dense, Flatten
from keras.optimizers import Adam


class DQNModel:
    def __init__(self, learning_rate, action_size, batch_size):
        self.learning_rate = learning_rate
        self.action_size = action_size
        self.model = self._build_model(batch_size)
        self.target_model = self._build_model(batch_size)
        self.update_target_model()

    def _build_model(self, batch_size):
        model = Sequential()

        #  Add convolutional and normalization layers
        model.add(Convolution2D(filters=32, kernel_size=8, strides=4, activation='relu'
                                , batch_input_shape=(None, 4, 84, 84), data_format='channels_first'))
        model.add(Convolution2D(filters=64, kernel_size=4, strides=2, activation='relu'))
        model.add(Convolution2D(filters=64, kernel_size=3, strides=1, activation='relu'))

        # make convolution layers falttend (1 dimensional)
        model.add(Flatten())
        #  add FC layers
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        # configure learning process
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=['accuracy'])

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())  # copy weights from model to target_model