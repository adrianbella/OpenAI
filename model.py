from keras import Input, Model
from keras.layers import Convolution2D
from keras.layers import Dense, Flatten
from keras import backend as K
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

        normalized = keras.layers.Lambda(lambda x: x / 255.0)(observations_input)

        #  Add convolutional and normalization layers
        hidden_conv_layer_1 = Convolution2D(filters=32, kernel_size=8, strides=4, activation='relu', data_format='channels_first')(normalized)
        hidden_conv_layer_2 = Convolution2D(filters=64, kernel_size=4, strides=2, activation='relu')(hidden_conv_layer_1)
        hidden_conv_layer_3 = Convolution2D(filters=64, kernel_size=3, strides=1, activation='relu')(hidden_conv_layer_2)
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
        model.compile(loss=DQNModel.huber_loss,
                      optimizer=keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.95, epsilon=0.01))

        return model

    @staticmethod
    def huber_loss(a, b, in_keras=True):
        error = a - b
        quadratic_term = error * error / 2
        linear_term = abs(error) - 1 / 2
        use_linear_term = (abs(error) > 1.0)
        if in_keras:
            # Cast the booleans to floats
            use_linear_term = K.cast(use_linear_term, 'float32')
        return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term

    def update_target_model(self):
        self.target_model = self.copy_model()  # copy weights from model to target_model

    def copy_model(self):
        self.model.save('model')
        return keras.models.load_model('model', custom_objects={'huber_loss': DQNModel.huber_loss})
