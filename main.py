import numpy as np
import cv2
import os
from tqdm import tqdm
import csv

from config import Config
from camera import Camera
from player_v2 import DetectPlayer, PlayerState
# from group_extractor import GroupExtractor
from kmeans import KMeansGroupExtractor
from tracedraw import TraceManager
from transform import bboxc2xywh, bboxc2feetxy, bboxc2center
from track_v2 import tracking
from ball import DetectBall, ball_tracking

def remove_lowconf_data(data, threshold):
    new_data = []
    for bboxc in data:
        if bboxc[2] > threshold:
            new_data.append(bboxc)
    return new_data

def get_max_conf_ball(data, threshold):
    new_data = []
    if data == []:
        return new_data
    max_thres_index = np.argmax([bb_pos[2] for bb_pos in data])
    [[x1, y1], [x2, y2], conf] = data[max_thres_index]

    if conf > threshold:
        new_data = [[[x1, y1], [x2, y2], conf]]
    return new_data

def remove_outside_people(data, camera):
    new_data = []
    feetxy = bboxc2feetxy(data)
    for bboxc, xy in zip(data, feetxy):
        if camera.in_field(xy):
            new_data.append(bboxc)
    return new_data

def remove_outside_ball(data, camera):
    new_data = []
    center = bboxc2center(data)
    for bboxc, xy in zip(data, center):
        if camera.in_field(xy):
            new_data.append(bboxc)
    return new_data

def players_data_to_csv(cfg, football, players, out_idx):

    if football.hold_group != None and out_idx == (cfg.RUN -1):
        if football.pass_count != 0 and football.pass_count in football.number_of_consecutive_passes[football.hold_group]:
            football.number_of_consecutive_passes[football.hold_group][football.pass_count] += 1
        player = next((player for player in players if player.number == football.hold_id), None)
        player.try_pass += 1
        player.pass_failure += 1

    with open(cfg.RECORD.CHECKFILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'{out_idx}'] + ['Players_data'] + [player.number for player in players])
        writer.writerow(['0'] + ['MaxSpeed'] + [f"{player.max_speed:.2f}" if player.state == PlayerState.Steady and player.max_speed else " " for player in players])
        writer.writerow(['0'] + ['a'] + [f"{player.acceleration:.2f}" if player.state == PlayerState.Steady and player.acceleration else " " for player in players])
        writer.writerow(['0'] + ['Distances'] + [f"{player.total_distance:.2f}" if player.state == PlayerState.Steady and player.total_distance else " " for player in players])
        writer.writerow(['0'] + ['pass_success'] + [player.pass_success for player in players])
        writer.writerow(['0'] + ['pass_failure'] + [player.pass_failure for player in players])
        writer.writerow(['0'] + ['pass_attempt'] + [player.try_pass for player in players])

        writer.writerow([f'{out_idx}'] + ['Group_data'] + ["Number of consecutive passes"])

        writer.writerow(['0'] + ['0'] + [i for i in range(1, 12)])
        writer.writerow(['0'] + ['Group0'] + [times for _, times in football.number_of_consecutive_passes[0].items()])
        writer.writerow(['0'] + ['Group1'] + [times for _, times in football.number_of_consecutive_passes[1].items()])

        writer.writerow([f'{out_idx}'] + ['Group_data'] + ["intercept"])
        writer.writerow(['0'] + ['Group0'] + [football.intercept_num[0]])
        writer.writerow(['0'] + ['Group1'] + [football.intercept_num[1]])

cfg = Config()
# group_extractor = GroupExtractor(cfg)
group_extractor = KMeansGroupExtractor(cfg)
draw = TraceManager(cfg)
cameras = []
# open(cfg.RECORD.CHECKFILE, 'w').close()

# set out video
cfg.mkdir()
out_fps = round(cfg.OUT_VIDEO.FPS)
out_resolution = (cfg.OUT_VIDEO.W, cfg.OUT_VIDEO.H)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
filepath = os.path.join(cfg.OUT_VIDEO.ROOT, cfg.OUT_VIDEO.FILENAME)
connect = cv2.VideoWriter(filepath, fourcc, out_fps, out_resolution)

players = []
football = None

for camera_info in cfg.CAMERAS:
    cameras.append(Camera(camera_info))

# low resolution
# connect
filepath_low = os.path.join(cfg.OUT_VIDEO.ROOT, cfg.OUT_VIDEO.FILENAME_LOW)
out_resolution_low = (int(cfg.OUT_VIDEO.W / 2), int(cfg.OUT_VIDEO.H / 2))
connect_low = cv2.VideoWriter(filepath_low, fourcc, out_fps, out_resolution_low)

# cameras
cameras_resolution = []
for i, camera in enumerate(cameras):
    out_video_path = f"camera{i+1}_{cfg.OUT_VIDEO.FILENAME_LOW}"
    filepath_camera_low = os.path.join(cfg.OUT_VIDEO.ROOT, out_video_path)
    out_camera_resolution = (int(camera.w / 2), int(camera.h / 2))
    cameras_resolution.append(cv2.VideoWriter(filepath_camera_low, fourcc, out_fps, out_camera_resolution))
# field_out
path = f"field_{cfg.OUT_VIDEO.FILENAME_LOW}"
field_filepath_low = os.path.join(cfg.OUT_VIDEO.ROOT, path)
field_resolution_low = (1150, 830)
field_low = cv2.VideoWriter(field_filepath_low, fourcc, out_fps, field_resolution_low)

open(cfg.RECORD.CHECKFILE, 'w').close()
with tqdm(total=cfg.RUN) as bar:
    for out_idx in range(0, cfg.RUN):
        ori_imgs = []
        detect_players = []
        detect_balls = []
        idxs = []
        for camera_idx, camera in enumerate(cameras):
            img, players_bboxc, ball_bboxc = camera.load_current_data(camera.start+out_idx)
            ori_imgs.append(img)
            idxs.append(camera.start+out_idx)

            ball_bboxc = get_max_conf_ball(ball_bboxc, cfg.THRESHOLD.Ball)
            ball_bboxc = remove_outside_ball(ball_bboxc, camera)
            detect_balls.append([DetectBall(cfg, xywh, camera_idx, cameras) for xywh in bboxc2xywh(ball_bboxc)])

            players_bboxc = remove_lowconf_data(players_bboxc, cfg.THRESHOLD.Person)
            players_bboxc = remove_outside_people(players_bboxc, camera)
            # labels, _, pds = group_extractor(img, players_bboxc)
            labels = group_extractor(img, players_bboxc, out_idx)
            detect_players.append([DetectPlayer(cfg, xywh, camera_idx, label, cameras, []) for xywh, label in zip(bboxc2xywh(players_bboxc), labels)])

        ####
        if out_idx == 0: # 0816
            y_coords = np.array([player.xywh[1] for player in detect_players[2]])
            min_y_index = np.argmin(y_coords)
            min_detection = detect_players[2][min_y_index]

            detect_players = [detect_players[0], [min_detection]]
        if 129 <= out_idx <= 327:
            detect_balls = [detect_balls[0], detect_balls[2]]
        if 1191 <= out_idx <= 1192:
            detect_balls = [detect_balls[0], detect_balls[1]]

        # if out_idx == 0: # 0809pass
        #     x_coords = np.array([player.xywh[0] for player in detect_players[0]])
        #     min_x_index = np.argmin(x_coords)
        #     camera1_detection = [player for index, player in enumerate(detect_players[0]) if index != min_x_index]

        #     detect_players = [camera1_detection, detect_players[1]]

        # if out_idx == 0: # 0809game2
        #     y_coords = np.array([player.xywh[1] for player in detect_players[1]])
        #     min_y_index = np.argmin(y_coords)
        #     min_detection = detect_players[1][min_y_index]

        #     detect_players = [detect_players[0], [min_detection]]

        # if out_idx == 0: # 0814
        #     first_person = [player for player in detect_players[0] if player.xywh[1] < 960]
        #     second_person = [player for player in detect_players[2] if player.xywh[1] < 600]
        #     detect_players = [first_person, second_person]
        ####

        players = tracking(cfg, players, detect_players)
        football = ball_tracking(cfg, football, detect_balls, players, out_idx)

        players.sort(key=lambda player: player.number)

        # draw
        imgs, field = draw.draw_on_ori(ori_imgs, players, football), draw.draw_on_field(players, football)
        imgs = draw.draw_detection(imgs, detect_players+detect_balls)
        canvas = draw.combine(imgs, field, out_idx, idxs)
        canvas = draw.put_pass_info(canvas, football)
        canvas = draw.put_speed_info(canvas, players)
        connect.write(canvas)

        # low resolution
        canvas_low = cv2.resize(canvas, (int(cfg.OUT_VIDEO.W / 2), int(cfg.OUT_VIDEO.H / 2)))
        connect_low.write(canvas_low)
        field_low.write(field)
        # cameras low resolution
        for camera_low, img in zip(cameras_resolution, imgs):
            canvas_low = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
            camera_low.write(canvas_low)

        if out_idx == 0:
            cv2.imwrite("first.jpg", canvas)

        cv2.imwrite(cfg.CHECKFRAME.ROOT + "/" + str(out_idx) + ".jpg", canvas)
        bar.update(1)
        players_data_to_csv(cfg, football, players, out_idx)


connect.release()
connect_low.release()
field_low.release()
for camera_low in cameras_resolution:
    camera_low.release()

# TODO: yolo-pose

