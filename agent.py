import random
import numpy as np
import time

from model import DQNModel
from buffer import RingBuffer


class DQNAgent:

    def __init__(self, env, action_size, config, batch_size):
        self.memory = RingBuffer(int(config.config_section_map()['memorysize']))
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

    def action(self, fi_t, env_sample):

        num_random = random.uniform(0, 1)

        if num_random <= self.epsilon:  # with probability epsilon do a random action
            return env_sample
        else:
            fi_t = np.expand_dims(fi_t, axis=0)
            action = self.dqn_model.model.predict([fi_t, np.ones([1, self.action_size])])
            return np.argmax(action[0])

    def replay(self, batch_size, csv_logger):

        states = np.zeros((batch_size, 4, 84, 84), dtype=float)
        actions = np.zeros((batch_size, 4), dtype=int)
        rewards = np.zeros((batch_size, 1), dtype=float)
        next_states = np.zeros((batch_size, 4, 84, 84), dtype=float)
        dones = np.ones((batch_size, 4), dtype=bool)

        mini_batch = random.sample(self.memory, batch_size)  # sample random mini_batch from D

        i = 0

        for state, action, reward, next_state, done in mini_batch:

            next_state = next_state.astype(float)
            state = state.astype(float)

            states[i] = state
            actions[i][action] = 1
            rewards[i] = reward
            next_states[i] = next_state
            dones[i] = [done, done, done, done]

            i += 1

        next_state_q_values = self.dqn_model.target_model.predict([next_states, np.ones(actions.shape)])

        next_state_q_values[dones] = 0

        Q_values = (rewards + self.gamma * np.max(next_state_q_values, axis=1))[0]

        #  Trains the model for a fixed number of epochs (iterations on a dataset)
        self.dqn_model.model.fit([states, actions], actions * Q_values[:, None],
                                 batch_size=batch_size, verbose=0, callbacks=[csv_logger])

    def load(self, name):
        self.dqn_model.model.load_weights(name)
        self.dqn_model.update_target_model()

    def save(self, name):
        self.dqn_model.model.save_weights(name)

    def decrease_epsilone(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
