import ConfigParser

class MyConfigParser:

    def __init__(self, section):
        self.section = section
        self.config = ConfigParser.ConfigParser()
        self.config.read("./config.ini")

    def config_section_map(self):
        dict1 = {}
        options = self.config.options(self.section)
        for option in options:
            try:
                dict1[option] = self.config.get(self.section, option)
            except:
                dict1[option] = None
        return dict1


