from keras import Sequential


class DQNAgent:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        return model