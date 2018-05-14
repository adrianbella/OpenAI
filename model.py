from keras import Input, Model
from keras.layers import Convolution2D
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import keras


class DQNModel:

    def __init__(self, learning_rate, action_size):
        self.learning_rate = learning_rate
        self.action_size = action_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):

        observations_input = Input((4, 84, 84), name='observations')
        actions_input = Input((self.action_size,), name='mask')

        #  Add convolutional and normalization layers
        hidden_layer_1 = Convolution2D(filters=32, kernel_size=8, strides=4, activation='relu', data_format='channels_first')(observations_input)
        hidden_layer_2 = Convolution2D(filters=64, kernel_size=4, strides=2, activation='relu')(hidden_layer_1)
        hidden_layer_3 = Convolution2D(filters=64, kernel_size=3, strides=1, activation='relu')(hidden_layer_2)

        # make convolution layers falttend (1 dimensional)
        flattened = Flatten()(hidden_layer_3)
        #  add FC layers
        hidden_layer_4 = Dense(512, activation='relu')(flattened)
        output_layer = Dense(self.action_size)(hidden_layer_4)

        filtered_output = keras.layers.merge([output_layer, actions_input], mode='mul')

        # configure learning process and initialize model
        model = Model(inputs=[observations_input, actions_input], output=filtered_output)
        model.compile(loss='mae', optimizer=Adam(lr=self.learning_rate))

        return model

    def update_target_model(self):
        self.target_model = self.copy_model(self.model)  # copy weights from model to target_model

    def copy_model(self, model):
        """Returns a copy of a keras model."""
        model.save('tmp_model')
        return keras.models.load_model('tmp_model')