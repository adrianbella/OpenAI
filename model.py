from keras import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense,Flatten
from keras import backend as K
from keras.optimizers import Adam


class DQNModel:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()



    def _build_model(self):
        model = Sequential()

        #Add convolutional and normalization layers
        model.add(Convolution2D(filters=16, kernel_size=8, strides=4, activation='relu', input_shape=(4, 84, 84), data_format='channels_first'))

        model.add(Convolution2D(filters=32, kernel_size=4, strides=2, activation='relu'))

        # reduce the num of parameters in our model by sliding a 2x2 pooling filter
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # make convolution layers falttend (1 dimensional)
        #add FC layers
        model.add(Dense(256, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        # configure learning process
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())  # copy weights from model to target_model

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)