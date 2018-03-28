import random
import numpy as np
from collections import deque
from model import DQNModel


class DQNAgent:
    def __init__(self, env, action_size):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.9  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.action_size = action_size
        self.env = env
        self.dqn_model = DQNModel(self.learning_rate)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def action(self, fi_t):

        num_random = random.uniform(0, 1)

        if num_random <= self.epsilon:  # with probability epsilon do a random action
            return random.randrange(self.action_size)
        else:
            fi_t = np.expand_dims(fi_t, axis=0)
            action = self.dqn_model.model.predict(fi_t)
            return np.argmax(action[0])

    def replay(self, batch_size, done):

        mini_batch = random.sample(self.memory, batch_size)  # sample random mini_batch from D

        for state, action, reward, next_state in mini_batch:

            next_state = np.expand_dims(next_state, axis=0)
            state = np.expand_dims(state, axis=0)

            if done:
                target = reward
            else:
                target = (reward + self.gamma * np.amax(self.dqn_model.target_model.predict(next_state)[0]))

            target_f = self.dqn_model.model.predict(state)
            target_f[0][action] = target
            #Trains the model for a fixed number of epochs (iterations on a dataset)
            self.dqn_model.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.dqn_model.model.load_weights(name)

    def save(self, name):
        self.dqn_model.model.save_weights(name)