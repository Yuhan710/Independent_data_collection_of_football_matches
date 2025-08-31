import cv2
import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from transform import bboxc2xywh

num = 0
class KMeansGroupExtractor(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_clusters = 2
        self.all_data_points = []
        filename = glob.glob(cfg.KmeansData_folder + "/" + "*.jpg")

        ### BGR
        data = [cv2.resize(cv2.imread(name), (20, 40), interpolation=cv2.INTER_AREA).reshape(-1) for name in filename]

        X = np.array(data)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')

        labels = self.kmeans.fit_predict(X)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        ###


        self.all_data_points.append([X_reduced, labels])

    def __call__(self, img, person_bboxc, out_idx):
        data = []
        global num

        if person_bboxc == []:
            return []

        for lu, rb, _ in person_bboxc:
            person_img = img[lu[1]:rb[1], lu[0]:rb[0]]
            # person_img = img[y+int(h*0.1):y+h-int(h*0.3)
            #                      , x+int(w*0.1):x+w-int(w*0.1)]
            # person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
            data.append(cv2.resize(person_img, (20, 40), interpolation=cv2.INTER_AREA).reshape(-1))
            # if out_idx % 90 == 0:
            #     cv2.imwrite(f'./kmeans_fit_data(R&Y&B)/{num}.jpg', person_img)
            #     num += 1
        X = np.array(data)

        # labels = self.kmeans.fit_predict(X)
        labels = self.kmeans.predict(X)
        centers = self.kmeans.cluster_centers_

        Team1color = list(map(int, self.cfg.GROUP.TEAM1.Color.split(',')))
        Team2color = list(map(int, self.cfg.GROUP.TEAM2.Color.split(',')))

        a_color, b_color = np.array(Team1color), np.array(Team2color)

        centers_reshaped = centers.reshape(2, -1, 3)
        centers_mean = np.mean(centers_reshaped, axis=1)

        new_labels = np.zeros_like(labels)

        if np.linalg.norm(centers_mean[0] - a_color) > np.linalg.norm(centers_mean[0] - b_color):
            new_labels[labels == 0] = 1
            new_labels[labels == 1] = 0
        else:
            new_labels = labels

        ######################################################
        # pca = PCA(n_components=2)
        # X_reduced = pca.fit_transform(X)
        # self.all_data_points.append([X_reduced, new_labels])

        return new_labels

    # def draw(self, out_idx):
    #     if len(self.all_data_points) > 3:
    #         plt.figure(figsize=(8, 6))
    #         for data, labels in self.all_data_points[:len(self.all_data_points)-3]:
    #             color = ['r','y']
    #             for cluster in range(self.n_clusters):
    #                 cluster_points = data[labels == cluster]
    #                 plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color[cluster])

    #         for data, labels in self.all_data_points[len(self.all_data_points)-3:]:
    #             color = ['b','k']
    #             for cluster in range(self.n_clusters):
    #                 cluster_points = data[labels == cluster]
    #                 plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color[cluster])

    #         # 添加標籤和標題
    #         plt.title("KMeans Clustering of Images")
    #         plt.xlabel("PCA Feature 1")
    #         plt.ylabel("PCA Feature 2")
    #         plt.legend()
    #         plt.grid(True)
    #         plt.savefig(f'kmeans_output/{out_idx}.png')
    #         plt.clf()




