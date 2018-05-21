from keras import Input, Model
from keras.layers import Convolution2D
from keras.layers import Dense, Flatten

import keras


class DQNModel:

    def __init__(self, learning_rate, action_size):
        self.learning_rate = learning_rate
        self.action_size = action_size
        self.model = self._build_model()
        self.target_model = self.copy_model()

    def _build_model(self):

        observations_input = Input((4, 84, 84), name='observations')
        actions_input = Input((self.action_size,), name='mask')

        normalized = keras.layers.Lambda(lambda x: x / 255.0)(observations_input)

        #  Add convolutional and normalization layers
        hidden_conv_layer_1 = Convolution2D(32, (8, 8), strides=(4, 4), padding='valid', activation='relu', data_format='channels_first')(normalized)
        hidden_conv_layer_2 = Convolution2D(64, (4, 4), strides=(2, 2), padding='valid', activation='relu')(hidden_conv_layer_1)
        hidden_conv_layer_3 = Convolution2D(64, (3, 3), strides=(1, 1), padding='valid' ,activation='relu')(hidden_conv_layer_2)
        # -------------------------------------------

        # make convolution layers falttend (1 dimensional)
        flattened = Flatten()(hidden_conv_layer_3)
        #  add FC layers
        hidden_fc_layer = Dense(512, activation='relu')(flattened)
        output_layer = Dense(self.action_size)(hidden_fc_layer)
        # -------------------------------------------

        filtered_output = keras.layers.multiply([output_layer, actions_input])

        # configure learning process and initialize model
        model = Model(inputs=[observations_input, actions_input], output=filtered_output)
        model.compile(loss='mse',
                      optimizer=keras.optimizers.RMSprop(lr=self.learning_rate))

        return model

    def update_target_model(self):
        self.target_model = self.copy_model()  # copy weights from model to target_model

    def copy_model(self):
        self.model.save('model')
        return keras.models.load_model('model', compile=False)
