import pandas as pd
import matplotlib.pyplot as plt
import datetime

class MyChart:

    config_section = ''
    date = datetime.date.today().strftime("%j")

    @staticmethod
    def paint_and_save_loss_chart(source):
        csv = pd.read_csv(source)
        df = pd.DataFrame(csv, columns=['loss'])

        df.plot()

        plt.savefig('./charts/losscharts/Day_' + MyChart.date + '_' + MyChart.config_section + '_' + 'losschart')
        plt.close()

    @staticmethod
    def paint_and_save_reward_chart(source):
        csv = pd.read_csv(source)
        df = pd.DataFrame(csv, columns=['rewards'])

        df.plot()

        plt.savefig('./charts/rewardcharts/Day_' + MyChart.date + '_' + MyChart.config_section + '_' + 'rewardchart')
        plt.close()


