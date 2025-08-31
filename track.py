import math
import numpy as np
import lap # type: ignore
from scipy.spatial.distance import cdist

from player import PlayerState

def same_people_process(cfg, players_set):
    new_player_set = []

    for players in players_set:
        for player in players:
            SamePersonFlag = False
            for idx, compare_players in enumerate(new_player_set):
                if any(math.sqrt((player.mapxy[0] - cmp_player.mapxy[0])**2 + (player.mapxy[1] - cmp_player.mapxy[1])**2) < cfg.THRESHOLD.Same_Person_distance for cmp_player in compare_players):
                    SamePersonFlag = True
                    new_player_set[idx].append(player)
                    break
            if not SamePersonFlag:
                new_player_set.append([player])

    # track_players = []
    # for player_set in new_player_set:
    #     if len(player_set) > 1:
    #         player_set = [player for player in players_set if player.state != PlayerState.Lost]

    #         camera_idx_set = {player.camera_idx for player in players_set}
    #         mapxys = np.array([player.mapxy for player in player_set])
    #         speed = np.array([player.speed for player in player_set])
    #         avg_mapxy = np.mean(mapxys, axis=0)
    #         avg_speed = np.mean(speed, axis=0)


    #     else:
    #         track_players.append(player_set[0])

    return new_player_set


def create_distance_array(players:list, detections:list):
    distance_array = np.zeros((len(players), len(detections)))
    for player_count, player in enumerate(players):
        player_mapx, player_mapy = player.mapxy[0], player.mapxy[1]
        for detection_count, detection in enumerate(detections):
            detection_mapx, detection_mapy = detection.mapxy[0], detection.mapxy[1]
            dx, dy = player_mapx-detection_mapx, player_mapy-detection_mapy
            distance = math.pow((math.pow(dx, 2) + math.pow(dy, 2)), 0.5)
            distance_array[player_count, detection_count] = distance
    return distance_array

def get_match(matrix, threshold):
    _, x, y = lap.lapjv(matrix, extend_cost=True, cost_limit=threshold)
    match_pairs = np.array([[ix, mx] for ix, mx in enumerate(x)])
    un_player_idx = np.where(x < 0)[0]
    un_detection_idx = np.where(y < 0)[0]
    return match_pairs, un_player_idx, un_detection_idx

def matching(players:list, detections:list, match_pairs:list):
    for player_index, detection_index in match_pairs:
        if player_index < 0 or detection_index < 0:
            continue
        new_xywh = detections[detection_index].xywh
        new_camera_idx = detections[detection_index].camera_idx
        new_label = detections[detection_index].group
        new_pd = detections[detection_index].pd

        players[player_index].to_steady(new_xywh, new_camera_idx, new_label, new_pd)
    return players

def tracking(cfg, players, detect_players):
    lost_players, steady_players, new_players, unmatch_detections = [], [], [], []

    if not players: # init
        for player in detect_players:
            player.to_steady(player.xywh, player.camera_idx, player.group, player.pd)
            new_players.append(player)
        return new_players

    for player in players:
        if player.state == PlayerState.Lost:
            lost_players.append(player)
        else:
            steady_players.append(player)
    distance_array = create_distance_array(steady_players, detect_players)

    if distance_array.size == 0:
        for detection in detect_players:
            if (len(players + new_players)) < cfg.MaxPlayerCount:
                detection.to_steady(detection.xywh, detection.camera_idx, detection.group, detection.pd)
                new_players.append(detection)
        return lost_players + steady_players + new_players

    match_pairs, unmatch_player_idx, unmatch_detections_idx = get_match(distance_array, threshold=60)
    steady_players = matching(steady_players, detect_players, match_pairs)
    _steady_players = []

    for sid in range(len(steady_players)):
        if sid in unmatch_player_idx:
            steady_players[sid].to_lost()
            lost_players.append(steady_players[sid])
        else:
            _steady_players.append(steady_players[sid])

    steady_players = _steady_players
    unmatch_detections = [detect_players[idx] for idx in unmatch_detections_idx]


    # ------ lost & unmatch_detections ------ #
    re_distance_array = create_distance_array(lost_players, unmatch_detections)

    if re_distance_array.size == 0:
        for detection in unmatch_detections:
            if (len(players)+len(new_players)) < cfg.MaxPlayerCount:
                detection.to_steady(detection.xywh, detection.camera_idx, detection.group, detection.pd)
                new_players.append(detection)
        return lost_players + steady_players + new_players


    rematch_pairs, unmatch_player_idx, unmatch_detections_idx = get_match(re_distance_array, threshold=200)
    lost_players = matching(lost_players, unmatch_detections, rematch_pairs)
    _lost_players = []
    for player in lost_players:
        if player.state == PlayerState.Steady:
            steady_players.append(player)
        else:
            player.to_lost()
            _lost_players.append(player)
    lost_players = _lost_players
    unmatch_detections = [unmatch_detections[idx] for idx in unmatch_detections_idx]

    for detection in unmatch_detections:
        if (len(players + new_players)) < cfg.MaxPlayerCount:
            detection.to_steady(detection.xywh, detection.camera_idx, detection.group, detection.pd)
            new_players.append(detection)
    return lost_players + steady_players + new_players
