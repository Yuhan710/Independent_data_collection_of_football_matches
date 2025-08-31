import cv2
import numpy as np
import matplotlib.pyplot as plt

from player import PlayerState
from ball import BallState
from transform import xywh2feetxy, xywh2center

class TraceManager(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.field_img = cv2.imread(cfg.FIELD.STRATEGY.PATH)
        # self.color_table = {
        #     0: (128,0,128),   #purple
        #     1: (255,0,0),     #blue
        #     2: (0,128,255),   #orange(light red)
        #     3: (255,255,255), #white
        #     4: (0,0,0),       #black
        #     5: (203,192,255), #pink
        #     6: (0,0,255),     #dark red
        #     7: (0,0,0),       #black
        #     8: (0,128,0),     #green
        #     9: (0,255,255)    #yellow
        # }
        self.color_table = {
            0: (0,0,255),      #red
            1: (0,255,255),    #yellow
            # 2: (0,0,0),
            "ball": (255,255,255),
            # 0: (0,0,255),        #red
            # 1: (255,255,255),    #white
        }
        self.imgs_x, self.imgs_y = cfg.PUT_POSITION.IMGS.x, cfg.PUT_POSITION.IMGS.y
        self.field_x, self.field_y = cfg.PUT_POSITION.FIELD.x, cfg.PUT_POSITION.FIELD.y
        self.ball_info_x, self.ball_info_y = cfg.PUT_POSITION.DATA.Ball.x, cfg.PUT_POSITION.DATA.Ball.y
        self.speed_info_x, self.speed_info_y = cfg.PUT_POSITION.DATA.Speed.x, cfg.PUT_POSITION.DATA.Speed.y

    def draw_on_ori(self, imgs, players, football):
        '''
            ori img traces item(player): [xywh, group]
        '''
        for player in players:
            if player.state == PlayerState.Steady:
                for camera_idx, value in player.detect_players.items():
                    traces = value["trace"]
                    if len(traces) > 1:
                        for idx in range(1, len(traces)):
                            color = self.color_table[traces[idx][1]] # group
                            feet_xy1 = xywh2feetxy(traces[idx-1][0])[0]
                            feet_xy2 = xywh2feetxy(traces[idx][0])[0]
                            cv2.line(imgs[camera_idx], (feet_xy1[0], feet_xy1[1]), (feet_xy2[0], feet_xy2[1]), color, 2)

                        last_point = (int(feet_xy2[0]), int(feet_xy2[1]))
                    else:
                        color = self.color_table[traces[0][1]]
                        feet_xy = xywh2feetxy(traces[0][0])[0]
                        last_point = (int(feet_xy[0]), int(feet_xy[1]))

                    cv2.circle(imgs[camera_idx], last_point, 11, color, -1)
                    cv2.putText(imgs[camera_idx], str(player.number), last_point, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        '''
            ori img traces item(ball): [xywh]
        '''
        if football:
            if football.hold_group != None:
                for camera_idx, value in football.detect_balls.items():
                    traces = value["trace"]
                    if len(traces) > 1:
                        for idx in range(1, len(traces)):
                            color = self.color_table["ball"] # group
                            center_xy1 = xywh2center(traces[idx-1][0])
                            center_xy2 = xywh2center(traces[idx][0])
                            cv2.line(imgs[camera_idx], (center_xy1[0], center_xy1[1]), (center_xy2[0], center_xy2[1]), color, 2)

                        last_point = (int(center_xy2[0]), int(center_xy2[1]))
                    else:
                        color = self.color_table["ball"]
                        center_xy = xywh2center(traces[0][0])
                        last_point = (int(center_xy[0]), int(center_xy[1]))

                    cv2.circle(imgs[camera_idx], last_point, 11, color, -1)

        return imgs

    def draw_on_field(self, players, football):
        '''
            field traces item: [mapxy, group]
        '''
        field = self.field_img.copy()
        for player in players:
            if player.state == PlayerState.Steady:
                traces = player.get_trace()
                if len(traces) > 1:
                    for idx in range(1, len(traces)):
                        color = self.color_table[traces[idx][1]] # group
                        mapxy1 = traces[idx-1][0]
                        mapxy2 = traces[idx][0]
                        cv2.line(field, (int(mapxy1[0]), int(mapxy1[1])), (int(mapxy2[0]), int(mapxy2[1])), color, 2)
                    last_point = (int(mapxy2[0]), int(mapxy2[1]))
                else:
                    color = self.color_table[traces[0][1]]
                    mapxy = traces[0][0]
                    last_point = (int(mapxy[0]), int(mapxy[1]))

                cv2.circle(field, last_point, 8, color, -1)
                cv2.putText(field, str(player.number), last_point, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

                # draw speed
                right_top = (last_point[0]+50, last_point[1])

                # if player.speed:
                #     cv2.putText(field, str(f'Speed: {int(player.speed)}km/h'), right_top, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if football:
            if football.hold_group != None:
                traces = football.get_trace()
                if len(traces) > 1:
                    for idx in range(1, len(traces)):
                        color = self.color_table["ball"] # group
                        mapxy1 = traces[idx-1][0]
                        mapxy2 = traces[idx][0]
                        cv2.line(field, (int(mapxy1[0]), int(mapxy1[1])), (int(mapxy2[0]), int(mapxy2[1])), color, 2)
                    last_point = (int(mapxy2[0]), int(mapxy2[1]))
                else:
                    color = self.color_table["ball"]
                    mapxy = traces[0][0]
                    last_point = (int(mapxy[0]), int(mapxy[1]))

                cv2.circle(field, last_point, 8, color, -1)

        return field

    def draw_detection(self, imgs, detections):
        for camera in detections:
            for draw_obj in camera:
                color = self.color_table[draw_obj.group]
                lu = (int(draw_obj.xywh[0]), int(draw_obj.xywh[1]))
                rb = (int(draw_obj.xywh[0]+draw_obj.xywh[2]), int(draw_obj.xywh[1]+draw_obj.xywh[3]))
                cv2.rectangle(imgs[draw_obj.camera_idx], lu, rb, color, 2)

        return imgs

    def combine(self, imgs, field, out_idx, idxs):
        canvas = np.zeros((self.cfg.OUT_VIDEO.H, self.cfg.OUT_VIDEO.W, 3),dtype=np.uint8)
        for img, put_x, put_y in zip(imgs, self.imgs_x, self.imgs_y):
            h, w, _ = img.shape
            canvas[put_y:put_y+h, put_x:put_x+w] = img

        field_h, field_w, _ = field.shape
        canvas[self.field_y:self.field_y+field_h, self.field_x:self.field_x+field_w] = field


        cv2.putText(canvas, str(out_idx), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        put_y = 130
        for idx in idxs:
            cv2.putText(canvas, str(idx), (20, put_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
            put_y += 70

        return canvas

    def put_speed_info(self, canvas, players):
        cv2.putText(canvas, "Speed information:", (self.speed_info_x, self.speed_info_y-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        img = np.zeros((600, 2000, 3), dtype=np.uint8)
        rows, cols = 4, self.cfg.MaxPlayerCount+2

        height, width, _ = img.shape

        cell_width = width // cols
        cell_height = height // rows

        for i in range(rows + 1):
            start_point = (0, i * cell_height)
            end_point = (width, i * cell_height)
            cv2.line(img, start_point, end_point, (255, 255, 255), 2)

        for i in range(cols + 1):
            start_point = (i * cell_width, 0)
            end_point = (i * cell_width, height)
            cv2.line(img, start_point, end_point, (255, 255, 255), 2)

        # write text
        cv2.putText(img, "MaxSpeed", (15, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "a", (15, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        cv2.putText(img, "Distance", (15, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        x_start, y_start = cell_width+40, 100
        for i, player in enumerate(players):
            cv2.putText(img, str(player.number), (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            if player.state == PlayerState.Steady:
                if player.max_speed:
                    cv2.putText(img, f"{player.max_speed:.2f}", (x_start, y_start+150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if player.acceleration:
                    cv2.putText(img, f"{player.acceleration:.2f}", (x_start, y_start+300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if player.total_distance:
                    cv2.putText(img, f"{player.total_distance:.2f}", (x_start, y_start+450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            x_start += cell_width

        cv2.putText(img, "km/hr", (x_start, y_start+150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, " ", (x_start, y_start+300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, " km ", (x_start, y_start+450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        canvas[self.speed_info_y:self.speed_info_y+height, self.speed_info_x:self.speed_info_x+width] = img

        return canvas

    def put_pass_info(self, canvas, football):
        img = np.zeros((600, 2000, 3), dtype=np.uint8)
        rows, cols = 3, 13

        height, width = 500, 2000

        cell_width = width // cols
        cell_height = height // rows

        cv2.putText(img, "Number of consecutive passes:", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        for i in range(rows + 1):
            height_start = 60
            start_point = (0, i * cell_height+height_start)
            end_point = (width, i * cell_height+height_start)
            cv2.line(img, start_point, end_point, (255, 255, 255), 2)

        for i in range(cols + 1):
            start_point = (i * cell_width, 60)
            end_point = (i * cell_width, 60+height)
            cv2.line(img, start_point, end_point, (255, 255, 255), 2)

        # write text
        cv2.putText(img, "Group0", (15, cell_height+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "Group1", (15, cell_height*2+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        x_start, y_start = cell_width+40, 140
        if football:
            for consecutive_passes, times in football.number_of_consecutive_passes[0].items():
                cv2.putText(img, str(consecutive_passes), (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                if times != 0:
                    cv2.putText(img, f"{times}", (x_start, y_start+cell_height), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                x_start += cell_width
            x_start = cell_width+40
            for consecutive_passes, times in football.number_of_consecutive_passes[1].items():
                if times != 0:
                    cv2.putText(img, f"{times}", (x_start, y_start+cell_height*2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                x_start += cell_width

            cv2.putText(img, "times", (x_start, cell_height+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "times", (x_start, cell_height*2+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        canvas[self.ball_info_y:self.ball_info_y+img.shape[0], self.ball_info_x:self.ball_info_x+img.shape[1]] = img

        if football.hold_group != None:
            cv2.putText(canvas, f"hold_group: {football.hold_group}", (500, 2500), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
            cv2.putText(canvas, f"hold_id: {football.hold_id}", (500, 2650), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
            cv2.putText(canvas, f"pass_count: {football.pass_count}", (1500, 2500), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
            cv2.putText(canvas, f"distance: {football.distance}", (1500, 2650), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
            if football.disturbed:
                cv2.putText(canvas, "disturbed", (2500, 2500), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        return canvas






