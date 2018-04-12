import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MyChart:

    loss = {}

    @staticmethod
    def add_new_loss(lossf):
        MyChart.loss = lossf
        print(lossf)

    @staticmethod
    def paint_result_chart():
        ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
        ts = ts.cumsum()
        ts.plot()