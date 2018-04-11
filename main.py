import gym as gym
import numpy as np
import datetime
import logging

from agent import DQNAgent
from state_builder import StateBuilder
from video import VideoRecorder
from config import MyConfigParser

if __name__ == "__main__":

    config = MyConfigParser('BreakoutV1')
    env = gym.make(config.config_section_map()['game'])
    action_size = env.action_space.n
    agent = DQNAgent(env, action_size, config)
    video = VideoRecorder()

    done = False
    batch_size = int(config.config_section_map()['batchsize'])
    EPISODES = int(config.config_section_map()['episodes'])

    logging.basicConfig(filename='./log/'+str(datetime.datetime.now())+'.log', level=logging.DEBUG)
    # agent.load("./save/breakout.h5")

    for i_episode in range(EPISODES):
        CNN_input_stack = np.zeros((5, 84, 84))  # initialization of the CNN input
        observation = env.reset()  # s1 = {x1}
        CNN_input_stack[0] = StateBuilder.pre_process(observation)
        action = env.action_space.sample()
        done = False
        sum_reward = 0

        for t in range(500000):

            env.render()

            video.record(observation)  # start video-recording

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

                fi_t = fi_t.astype('uint8')
                fi_t1 = fi_t1.astype('uint8')

                agent.remember(fi_t, action, reward, fi_t1)  # Store transition (fi_t,a_t,r_t,fi_t+1) in D

                if len(agent.memory) > batch_size:  # Sample random minibatch of transitions() from D
                    agent.replay(batch_size, done)

            else:
                CNN_input_stack[(t + 1)] = StateBuilder.pre_process(observation)

            if done:
                logging.info("episode: {}/{}, score: {}, epsilone: {:.2}, timestep: {}"
                      .format(i_episode, EPISODES, sum_reward, agent.epsilon, t))
                print("episode: {}/{}, score: {}, epsilone: {:.2}, timestep: {}"
                      .format(i_episode, EPISODES, sum_reward, agent.epsilon, t))
                break

        agent.dqn_model.update_target_model()  # update target model every C step

        if i_episode % 100 == 0:
            filename = "./save/breakout_"+str(datetime.datetime.now())+".h5"
            agent.save(filename)

    video.stop()  # stop video-recording

