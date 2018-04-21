import gym as gym
import numpy as np

from pathlib import Path

from agent import DQNAgent
from state_builder import StateBuilder
from video import VideoRecorder
from config import MyConfigParser
from csvhandler import MyCSVHandler
from logger import MyLogger


def init():

    # Load saved Weigths if exits
    my_file = Path("./save/" + config.section + ".h5")
    if my_file.is_file():
        agent.load("./save/" + config.section + ".h5")
    #  ---------------------------------------------------

def check_if_do_nothing(last_30_actions):
    for i in last_30_actions:
        if i > 0:
            return False
    return True

if __name__ == "__main__":

    done = False
    max_reward = 0
    last_ten_rewards = []
    frame_count = 0

    config = MyConfigParser('BreakoutV1')
    init()
    env = gym.make(config.config_section_map()['game'])
    action_size = env.action_space.n

    agent = DQNAgent(env, action_size, config)
    video = VideoRecorder(config.section)

    batch_size = int(config.config_section_map()['batchsize'])
    EPISODES = int(config.config_section_map()['episodes'])

    csv_handler = MyCSVHandler(config)

    logger = MyLogger(config.section)
    logger.log_config_parameters("./config.ini")# log the used parameter

    for i_episode in range(EPISODES):
        CNN_input_stack = np.zeros((5, 84, 84))  # initialization of the CNN input
        observation = env.reset()  # s1 = {x1}
        CNN_input_stack[0] = StateBuilder.pre_process(observation)
        action = env.action_space.sample()

        last_30_actions = []
        last_30_actions.append(action)

        done = False
        sum_reward = 0

        for t in range(500000):

            env.render()

            video.record(observation)  # start video-recording

            if t >= 4:

                fi_t = CNN_input_stack[0:4]  # fi_t = fi(s_t)

                #  check if last 30 actions was 0 and do random action if true
                if(len(last_30_actions) == 30 and check_if_do_nothing(last_30_actions)):
                    action = env.action_space.sample()
                    last_30_actions = []
                else:
                    action = agent.action(fi_t)
                    last_30_actions.append(action)
                # --------------------------------------------
                observation, reward, done, info = env.step(action)


                sum_reward += reward

                if(sum_reward > max_reward):
                    max_reward = sum_reward

                CNN_input_stack[0] = CNN_input_stack[1]  # overlap
                CNN_input_stack[1] = CNN_input_stack[2]
                CNN_input_stack[2] = CNN_input_stack[3]
                CNN_input_stack[3] = CNN_input_stack[4]
                CNN_input_stack[4] = StateBuilder.pre_process(observation)

                fi_t1 = CNN_input_stack[1:5]  # fi_t+1 = fi(s_t+1)

                agent.remember(fi_t, action, reward, fi_t1, done)  # Store transition (fi_t,a_t,r_t,fi_t+1) in D

                frame_count += 1

                #  update target model every C = 4000 frames
                if frame_count % 4000 == 0:
                    agent.dqn_model.update_target_model()
                # ------------------------------------------

                #  start experience replay after 50000 frames
                if frame_count > 50000 and t % 4 == 0:
                    agent.replay(batch_size, done, csv_handler.csv_file_handler, logger)
                # ------------------------------------------

            else:
                CNN_input_stack[(t + 1)] = StateBuilder.pre_process(observation)

            if done:

                logger.log_episode(i_episode, EPISODES, sum_reward, agent.epsilon, t, max_reward)
                csv_handler.write_csv_file(csv_handler.reward_file_path, int(sum_reward))
                break

        last_ten_rewards.append(sum_reward)

        # log average reached rewards after every 10 episodes
        if len(last_ten_rewards) == 10:
            avg = 0
            for i in last_ten_rewards:
                avg += i

            logger.log_10_avg((avg / len(last_ten_rewards)))
            last_ten_rewards = []
        # ----------------------------------------------------

        # save weights after every 10 episodes
        if i_episode % 10 == 0:
            filename = "./save/"+config.section+".h5"
            agent.save(filename)
        # -----------------------------------------------------

    video.stop()  # stop video-recording


