import gym as gym
import numpy as np
import datetime
import logging

from agent import DQNAgent
from state_builder import StateBuilder

EPISODES = 1000

if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    action_size = env.action_space.n
    agent = DQNAgent(env, action_size)
    done = False
    batch_size = 32
    logging.basicConfig(filename='./log/'+str(datetime.datetime.now())+'.log', level=logging.DEBUG)
    # agent.load("./save/breakout.h5")

    for i_episode in range(EPISODES):
        CNN_input_stack = np.zeros((5, 84, 84))  # initialization of the CNN input
        observation = env.reset()  # s1 = {x1}
        CNN_input_stack[0] = StateBuilder.pre_process(observation)
        action = env.action_space.sample()
        done = False
        sum_reward = 0

        for t in range(5000):

            env.render()

            if t >= 4:

                fi_t = CNN_input_stack[0:4]  # fi_t = fi(s_t)

                action = agent.action(fi_t)
                observation, reward, done, info = env.step(action)

                sum_reward += reward

                CNN_input_stack[0] = CNN_input_stack[1]  # overlap
                CNN_input_stack[1] = CNN_input_stack[2]
                CNN_input_stack[2] = CNN_input_stack[3]
                CNN_input_stack[3] = CNN_input_stack[4]
                CNN_input_stack[4] = StateBuilder.pre_process(observation)

                fi_t1 = CNN_input_stack[1:5]  # fi_t+1 = fi(s_t+1)

                agent.remember(fi_t, action, reward, fi_t1)  # Store transition (fi_t,a_t,r_t,fi_t+1) in D

            else:
                CNN_input_stack[(t + 1)] = StateBuilder.pre_process(observation)

            if done:
                logging.info("episode: {}/{}, score: {}, epsilone: {:.2}, timestep: {}"
                      .format(i_episode, EPISODES, sum_reward, agent.epsilon, t))
                print("episode: {}/{}, score: {}, epsilone: {:.2}, timestep: {}"
                      .format(i_episode, EPISODES, sum_reward, agent.epsilon, t))
                break

        agent.dqn_model.update_target_model()  # update target model every C step

        if len(agent.memory) > batch_size:
            agent.replay(batch_size, done)

        if i_episode % 100 == 0:
            filename = "./save/breakout_"+str(datetime.datetime.now())+".h5"
            agent.save(filename)

