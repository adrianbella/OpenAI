import gym as gym
import numpy as np

from DQNagent import DQNAgent
from StateBuilder import StateBuilder

EPISODES = 1000

if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    agent = DQNAgent(env)
    sb = StateBuilder()
    done = False
    batch_size = 32
    # agent.load("./save/breakout.h5")
    #TODO:initialize action-value function with random weights teta
    #TODO:initialize target action-value function with weights teta- = teta

    for i_episode in range(EPISODES):
        CNN_input_stack = np.zeros((5, 84, 84))  # initialization of the CNN input
        observation = env.reset()  # s1 = {x1}
        CNN_input_stack[0] = sb.preprocess(observation)
        action = env.action_space.sample()

        for t in range(500):

            env.render()


            if t >= 4:

                fi_t = CNN_input_stack[0:4]  # fi_t = fi(s_t)

                action = agent.action(fi_t)
                observation, reward, done, info = env.step(action)

                CNN_input_stack[0] = CNN_input_stack[1]  # overlap
                CNN_input_stack[1] = CNN_input_stack[2]
                CNN_input_stack[2] = CNN_input_stack[3]
                CNN_input_stack[3] = CNN_input_stack[4]
                CNN_input_stack[4] = sb.preprocess(observation)

                fi_t1 = CNN_input_stack[1:5]  # fi_t+1 = fi(s_t+1)

                agent.remember(fi_t, action, reward, fi_t1)

            else:
                CNN_input_stack[(t + 1)] = sb.preprocess(observation)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                print(info)
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if i_episode % 10 == 0:
           agent.save("./save/breakout.h5")

