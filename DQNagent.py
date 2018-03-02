import random
import gym as gym
import numpy as np
from collections import deque
from scipy.misc import imread, imsave, imresize

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
        self.numpyMatrix = np.zeros((84, 84))

    def preprocess(self,  observation):

        observation_gray_cropped = np.zeros((158, 144))
        observation_gray = observation * [0.299, 0.587, 0.114]  # gray-scale

        notNullRGBIndexArray = observation_gray.nonzero()
        length = notNullRGBIndexArray[0].__len__()

        for i in range(0, length, 3):  # minden 4. index veszunk csak, mert azok reprezentalnak egy uj pixelt
            x = notNullRGBIndexArray[1][i]
            y = notNullRGBIndexArray[0][i]

            if (31 < y < 190 and 7 < x < 152):#cropping
                observation_gray_cropped[(y - 32)][(x - 8)] = observation_gray[y][x][0]+observation_gray[y][x][1]+observation_gray[y][x][2]

        self.numpyMatrix = imresize(observation_gray_cropped,(84, 84)) #down-scale

        return self.numpyMatrix

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
        DQNinput = np.zeros((4, 84, 84))
        observation = env.reset()
        DQNinput[0] = agent.preprocess(observation)

        for t in range(500):
            action = agent.getAction()
            env.render()
            observation,reward, done, info = env.step(action)


            if t >= 3:
                DQNinput[0] = DQNinput[1]
                DQNinput[1] = DQNinput[2]
                DQNinput[2] = DQNinput[3]
                DQNinput[3] = agent.preprocess(observation)
                fi = DQNinput
            else:
                DQNinput[(t + 1) % 4] = agent.preprocess(observation)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                print(info)
                break

