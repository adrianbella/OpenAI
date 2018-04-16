import gym as gym
import numpy as np
import datetime
import logging

from pathlib import Path
from agent import DQNAgent
from state_builder import StateBuilder
from video import VideoRecorder
from config import MyConfigParser
from chart import MyChart
from csvhandler import MyCSVHandler

from keras.callbacks import CSVLogger

if __name__ == "__main__":

    config = MyConfigParser('BreakoutV1')
    MyChart.config_section = config.section
    env = gym.make(config.config_section_map()['game'])
    action_size = env.action_space.n
    agent = DQNAgent(env, action_size, config)
    video = VideoRecorder(config.section)

    done = False
    max_reward = 0
    last_ten_rewards = []

    batch_size = int(config.config_section_map()['batchsize'])
    EPISODES = int(config.config_section_map()['episodes'])

    csv_file_str_loss = './csvfiles/lossfunctions/' + 'Day_' + datetime.date.today().strftime("%j") + '_' + config.section + '_' + '_training_loss.csvfiles'
    csv_logger = CSVLogger(csv_file_str_loss, append=True)

    csv_file_str_reward = './csvfiles/rewardfunctions/' + 'Day_' + datetime.date.today().strftime("%j") + '_' + config.section + '_' + '_training_reward.csvfiles'
    MyCSVHandler.write_cvs_file_title(csv_file_str_reward, 'rewards')

    logging.basicConfig(filename='./log/' + str(datetime.datetime.now()) + '.log', level=logging.DEBUG)
    config.config_log_parameters(logging)  # log the used parameters

    my_file = Path("./save/"+config.section+".h5")
    if my_file.is_file():
        agent.load("./save/"+config.section+".h5")

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

                if(sum_reward > max_reward):
                    max_reward = sum_reward

                CNN_input_stack[0] = CNN_input_stack[1]  # overlap
                CNN_input_stack[1] = CNN_input_stack[2]
                CNN_input_stack[2] = CNN_input_stack[3]
                CNN_input_stack[3] = CNN_input_stack[4]
                CNN_input_stack[4] = StateBuilder.pre_process(observation)

                fi_t1 = CNN_input_stack[1:5]  # fi_t+1 = fi(s_t+1)

                fi_t = fi_t.astype('uint8')
                fi_t1 = fi_t1.astype('uint8')

                agent.remember(fi_t, action, reward, fi_t1)  # Store transition (fi_t,a_t,r_t,fi_t+1) in D

                if len(agent.memory) > batch_size and t % 4 == 0:
                    agent.replay(batch_size, done, csv_logger)

            else:
                CNN_input_stack[(t + 1)] = StateBuilder.pre_process(observation)

            if done:
                logging.info("episode: {}/{}, score: {}, epsilone: {:.2}, timestep: {}, maximum reward: {}"
                      .format(i_episode, EPISODES, sum_reward, agent.epsilon, t, max_reward))

                print("episode: {}/{}, score: {}, epsilone: {:.2}, timestep: {}, maximum reward: {}"
                      .format(i_episode, EPISODES, sum_reward, agent.epsilon, t, max_reward))

                MyChart.paint_and_save_loss_chart(csv_file_str_loss)

                MyCSVHandler.write_csv_file(csv_file_str_reward, int(sum_reward))
                MyChart.paint_and_save_reward_chart(csv_file_str_reward)

                break

        agent.dqn_model.update_target_model()  # update target model every C step

        last_ten_rewards.append(sum_reward)

        if len(last_ten_rewards) == 10:
            avg = 0

            for i in last_ten_rewards:
                avg += i

            avg = avg / len(last_ten_rewards)

            logging.info("Average of the last 10 episodes: {}".format(avg))
            print ("Average of the last 10 episodes: {}".format(avg))

            last_ten_rewards = []

        if i_episode % 10 == 0:
            filename = "./save/"+config.section+".h5"
            agent.save(filename)

    video.stop()  # stop video-recording

