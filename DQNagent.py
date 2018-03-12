import random
import numpy as np
from collections import deque

import DQNmodel as model

class DQNAgent:
    def __init__(self,env):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.9  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.env = env

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def action(self, fi_t):

        num_random = random.uniform(0, 1)

        if num_random <= self.epsilon:
            return self.env.action_space.sample()

        action = model.predict(fi_t)

        return np.argmax(action[0])



    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                target = (reward + self.gamma * np.amax(model.predict(next_state)[0]))

            target_f = model.predict(state)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        model.load_weights(name)

    def save(self, name):
        model.save_weights(name)