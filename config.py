import yaml
import argparse
import easydict

from pathWrapper import mk_dir

class Config(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--config', type=str, required=True)
        self.args = self.parser.parse_args()

        with open(self.args.config) as yaml_file:
            data = easydict.EasyDict(yaml.load(yaml_file, Loader=yaml.SafeLoader))

        self.CAMERAS = data.CAMERAS
        self.OUT_VIDEO = data.OUT_VIDEO
        self.RUN = data.RUN
        self.PUT_POSITION = data.PUT_POSITION
        self.REID = data.REID
        self.LEN = data.LEN
        self.FIELD = data.FIELD
        self.RECORD = data.RECORD
        self.THRESHOLD = data.THRESHOLD
        self.GROUP = data.GROUP
        self.MaxPlayerCount = data.MaxPlayerCount
        self.KmeansData_folder = data.KmeansData_folder
        self.CHECKFRAME = data.CHECKFRAME


    def mkdir(self):
        mk_dir(self.OUT_VIDEO.ROOT)
        mk_dir(self.RECORD.ROOT)
        mk_dir(self.CHECKFRAME.ROOT)
