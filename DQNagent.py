import random
import gym as gym
import numpy as np
from collections import deque

EPISODES = 1000


class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.9  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model
        self.numpymatrix = np.zeros((210,160))


    def preprocess(self,  observation):
        myindexarray = observation.nonzero()# elmentjuk azokat az indexeket egy tombbe,ahol nem csupa 0 az RGB kod

        for i in range(0, np.count_nonzero(myindexarray) / 4 , 4):# minden 4. index veszunk csak, mert azok reprezentalnak egy uj pixelt
                self.numpymatrix[myindexarray[0][i]][myindexarray[1][i]] = observation[myindexarray[0][i]][myindexarray[1][i]][0] * 0.299\
                                         + observation[myindexarray[0][i]][myindexarray[1][i]][1] * 0.587\
                                         + observation[myindexarray[0][i]][myindexarray[1][i]][2] * 0.114
        return self.numpymatrix

    def _build_model(self):
        model = 1
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def getAction(self):
        return  env.action_space.sample()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    agent = DQNAgent()
    done = False

    for i_episode in range(EPISODES):
        observation = env.reset()
        fi = agent.preprocess(observation)

        for t in range(1000):
            action = agent.getAction()
            env.render()
            observation,reward, done, info = env.step(action)
            fi = agent.preprocess(observation)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                print(info)
                break
