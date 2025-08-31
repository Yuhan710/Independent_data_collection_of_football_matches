import cv2
import json
import numpy as np

class Camera(object):
    def __init__(self, camera):
        self.cap = cv2.VideoCapture(camera.PATH.ORI_Video)
        with open(camera.PATH.JSON, 'r') as file:
            self.json_file = json.load(file)
        self.w = camera.W
        self.h = camera.H
        self.start, self.end = camera.START, camera.END

        mask = cv2.imread(camera.PATH.MASK, cv2.IMREAD_GRAYSCALE)
        mask[mask > 100] = 255 ; mask[mask != 255] = 0
        cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.mask_points = cnts[0]

        self.homo_matrix, _ = cv2.findHomography(np.array(camera.Src_Point), np.array(camera.Dst_Point))

    def load_current_data(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, img = self.cap.read()

        data = self.json_file[str(frame_idx)]
        person_data = data["person"] if "person" in data else []
        ball_data = data["sports ball"] if "sports ball" in data else []

        return img, person_data, ball_data

    def in_field(self, feet_xy):
        return True if cv2.pointPolygonTest(self.mask_points, (feet_xy[0], feet_xy[1]), 1) > 0 else False




