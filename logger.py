import logging
import ConfigParser
import datetime

class MyLogger:

    def __init__(self, section):

        self.section = section
        self.config = ConfigParser.ConfigParser()
        logging.basicConfig(filename='./log/' + str(datetime.datetime.now()) + '.log', level=logging.DEBUG)

    def log_config_parameters(self, file):
        dict1 = {}
        self.config.read(file)
        options = self.config.options(self.section)
        logging.info("Section: {}".format(self.section))
        for option in options:
            try:
                logging.info("\t\t {} : {}".format(option, self.config.get(self.section, option)))
            except:
                dict1[option] = None

    def log_episode(self, i_episode, EPISODES, sum_reward, epsilon, t, max_reward):

        logging.info("episode: {}/{}, score: {}, epsilone: {:.2}, timestep: {}, maximum reward: {}"
                     .format(i_episode, EPISODES, sum_reward, epsilon, t, max_reward))

        print("episode: {}/{}, score: {}, epsilone: {:.2}, timestep: {}, maximum reward: {}"
              .format(i_episode, EPISODES, sum_reward, epsilon, t, max_reward))

    def log_10_avg(self, avg):
        logging.info("Average of the last 10 episodes: {}".format(avg))
        print ("Average of the last 10 episodes: {}".format(avg))

    def log_replay(self, target, a, t, done):
        logging.info("\t target: {}, a: {}, t:{} , done: {}".format(target, a, t, done))
        print ("\t target: {}, a: {}, t:{} , done: {}".format(target, a, t, done))