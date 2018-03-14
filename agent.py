import random
import numpy as np
from collections import deque
from model import DQNModel


class DQNAgent:
    def __init__(self, env):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.9  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.env = env
        self.dqn_model = DQNModel(self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, fi_t):

        num_random = random.uniform(0, 1)

        if num_random <= self.epsilon:
            return self.env.action_space.sample()

        action = self.dqn_model.model.predict(fi_t)

        return np.argmax(action[0])

    def replay(self, batch_size):  #TODO: replace target and simple model to the right places

        mini_batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in mini_batch:
            target = reward

            if not done:
                target = (reward + self.gamma * np.amax(self.dqn_model.model.Sequential.predict(next_state)[0]))

            target_f = self.dqn_model.model.Sequential.predict(state)
            target_f[0][action] = target
            self.dqn_model.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.dqn_model.model.load_weights(name)

    def save(self, name):
        self.dqn_model.model.save_weights(name)