import random
import numpy as np
from collections import deque
from model import DQNModel


class DQNAgent:

    def __init__(self, env, action_size, config):
        self.memory = deque(maxlen=int(config.config_section_map()['memorysize']))
        self.gamma = float(config.config_section_map()['gamma'])    # discount rate
        self.epsilon = float(config.config_section_map()['epsilon'])  # exploration rate
        self.epsilon_min = float(config.config_section_map()['epsilonmin'])
        self.epsilon_decay = float(config.config_section_map()['epsilondecay'])
        self.learning_rate = float(config.config_section_map()['learningrate'])
        self.action_size = action_size
        self.env = env
        self.dqn_model = DQNModel(self.learning_rate, action_size)

    def remember(self, state, action, reward, next_state, done):
        state = state.astype('uint8')
        next_state = next_state.astype('uint8')

        self.memory.append((state, action, reward, next_state, done))

    def action(self, fi_t):

        num_random = random.uniform(0, 1)

        if num_random <= self.epsilon:  # with probability epsilon do a random action
            return random.randrange(self.action_size)
        else:
            fi_t = np.expand_dims(fi_t, axis=0)
            action = self.dqn_model.model.predict_on_batch(fi_t)
            return np.argmax(action[0])

    def replay(self, batch_size, done, csv_logger):

        mini_batch = random.sample(self.memory, batch_size)  # sample random mini_batch from D

        for state, action, reward, next_state, done in mini_batch:

            next_state = np.expand_dims(next_state, axis=0)
            state = np.expand_dims(state, axis=0)

            next_state = next_state.astype(float)
            state = state.astype(float)

            target = self.dqn_model.model.predict_on_batch(state)

            if done:
                target[0][action] = reward
            else:
                target_Q = self.dqn_model.target_model.predict_on_batch(next_state)
                target[0][action] = reward + self.gamma * np.amax(target_Q)

            #Trains the model for a fixed number of epochs (iterations on a dataset)
            self.dqn_model.model.fit(state, target, epochs=1, verbose=0, callbacks=[csv_logger])

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def load(self, name):
        self.dqn_model.model.load_weights(name)
        self.dqn_model.update_target_model()

    def save(self, name):
        self.dqn_model.model.save_weights(name)