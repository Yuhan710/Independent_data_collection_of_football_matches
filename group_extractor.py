import torch
import cv2

from REID.torchreid.utils import FeatureExtractor

num = 1

class GroupExtractor(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.extractor = FeatureExtractor(
            model_name=cfg.REID.ModelName,
            num_classes=cfg.REID.NUM_Class,
            model_path=cfg.REID.Pretrain,
            device='cuda'
        )

    def __call__(self, img, person_bboxc):
        global num
        labels, features, pds = [], [], []
        for idx in range(len(person_bboxc)):
            label, feature = self.extractor(img[person_bboxc[idx][0][1]:person_bboxc[idx][1][1], person_bboxc[idx][0][0]:person_bboxc[idx][1][0]])
            ###
            softmaxed_label = torch.softmax(label, dim=1)
            pds.append(softmaxed_label.squeeze().tolist())
            ###
            label = torch.argmax(torch.softmax(label, dim=1), dim=1)
            labels.append(label.item()) ; features.append(feature.cpu().numpy())

        return labels, features, pds
