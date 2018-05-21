import csv
from keras.callbacks import CSVLogger
import datetime


class MyCSVHandler:

    def __init__(self, config):

        self.loss_file_path = './csvfiles/lossfunctions/' + 'Day_' + datetime.date.today().strftime('%j') + '_' + config.section + '_' + '_training_loss.csvfiles'
        self.reward_file_path = './csvfiles/rewardfunctions/' + 'Day_' + datetime.date.today().strftime('%j') + '_' + config.section + '_' + '_training_reward.csvfiles'
        self.q_values = './csvfiles/qvalues/' + 'Day_' + datetime.date.today().strftime('%j') + '_' + config.section + '_' + '_training_q_values.csvfiles'

        self.csv_file_handler = CSVLogger(self.loss_file_path, append=True)
        self.write_cvs_file_title(self.reward_file_path, 'rewards')

    def write_csv_file(self, source, data):

        with open(source, 'a') as f:
            writer = csv.writer(f, delimiter=' ', quotechar='|')
            writer.writerow([data])
            f.close()

    def write_cvs_file_title(self, source, title):
        with open(source, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([title])
            f.close()

    def get_csv_logger(self):
        return self.csv_file_handler

    def write_q_values(self, data):
        with open(self.q_values, 'a') as f:
            writer = csv.writer(f, delimiter=' ', quotechar='|')
            writer.writerow([data])
            f.close()
