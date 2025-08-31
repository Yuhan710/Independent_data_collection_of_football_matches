import cv2
import glob
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from transform import bboxc2xywh

def white_balance(img):
    b, g, r = cv2.split(img)
    m, n = b.shape

    I_r_2 = np.zeros(r.shape)
    I_b_2 = np.zeros(b.shape)
    sum_I_r_2, sum_I_r, sum_I_b_2, sum_I_b, sum_I_g = 0, 0, 0, 0, 0
    max_I_r_2, max_I_r, max_I_b_2, max_I_b, max_I_g = int(r[0][0] ** 2), int(r[0][0]), int(b[0][0] ** 2), int(b[0][0]), int(g[0][0])
    for i in range(m):
        for j in range(n):
            I_r_2[i][j] = int(r[i][j] ** 2)
            I_b_2[i][j] = int(b[i][j] ** 2)
            sum_I_r_2 = I_r_2[i][j] + sum_I_r_2
            sum_I_b_2 = I_b_2[i][j] + sum_I_b_2
            sum_I_g = g[i][j] + sum_I_g
            sum_I_r = r[i][j] + sum_I_r
            sum_I_b = b[i][j] + sum_I_b
            if max_I_r < r[i][j]:
                max_I_r = r[i][j]
            if max_I_r_2 < I_r_2[i][j]:
                max_I_r_2 = I_r_2[i][j]
            if max_I_g < g[i][j]:
                max_I_g = g[i][j]
            if max_I_b_2 < I_b_2[i][j]:
                max_I_b_2 = I_b_2[i][j]
            if max_I_b < b[i][j]:
                max_I_b = b[i][j]

    [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
    [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])

    b0, g0, r0 = np.zeros(b.shape, np.uint8), np.zeros(g.shape, np.uint8), np.zeros(r.shape, np.uint8)
    for i in range(m):
        for j in range(n):
            b_point = u_b * (b[i][j] ** 2) + v_b * b[i][j]
            g0[i][j] = g[i][j]
            r_point = u_r * (r[i][j] ** 2) + v_r * r[i][j]
            if r_point>255:
                r0[i][j] = 255
            else:
                if r_point<0:
                    r0[i][j] = 0
                else:
                    r0[i][j] = r_point
            if b_point>255:
                b0[i][j] = 255
            else:
                if b_point<0:
                    b0[i][j] = 0
                else:
                    b0[i][j] = b_point
    return cv2.merge([b0, g0, r0])

num = 0
class KMeansGroupExtractor(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_clusters = 2
        self.all_data_points = []
        ##Use image with only middle part
        filename = glob.glob(cfg.KmeansData_folder + "/" + "*.jpg")

        ### White balance HSV + HSV
        data = []
        with tqdm(total=len(filename)) as bar:
            for name in filename:
                kmeans_img = cv2.imread(name)
                hsv_image = cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2HSV)
                white_balance_img = white_balance(kmeans_img)

                hsv_white_balance = cv2.cvtColor(white_balance_img, cv2.COLOR_BGR2HSV)

                hsv_image_flat = cv2.resize(hsv_image, (20, 40), interpolation=cv2.INTER_AREA).reshape(-1)
                hsv_white_balance_flat = cv2.resize(hsv_white_balance, (20, 40), interpolation=cv2.INTER_AREA).reshape(-1)
                kmeans_image_flatten = np.dstack((hsv_white_balance_flat, hsv_image_flat)).reshape(-1)

                data.append(kmeans_image_flatten)
                bar.update(1)

        X = np.array(data)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')

        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        labels = self.kmeans.fit_predict(X_reduced)
        ###


        self.all_data_points.append([X_reduced, labels])

    def __call__(self, img, person_bboxc, out_idx):
        person_data = []
        global num

        if person_bboxc == []:
            return []

        for lu, rb, _ in person_bboxc:
            x = lu[0]
            y = lu[1]
            w = rb[0] - lu[0]
            h = rb[1] - lu[1]
            ### mid
            person_img = img[y+int(h*0.3):y+h-int(h*0.4)
                                 , x+int(w*0.3):x+w-int(w*0.3)]

            hsv_image = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
            white_balance_img = white_balance(person_img)

            hsv_white_balance = cv2.cvtColor(white_balance_img, cv2.COLOR_BGR2HSV)

            hsv_image_flat = cv2.resize(hsv_image, (20, 40), interpolation=cv2.INTER_AREA).reshape(-1)
            hsv_white_balance_flat = cv2.resize(hsv_white_balance, (20, 40), interpolation=cv2.INTER_AREA).reshape(-1)
            person_img_flatten = np.dstack((hsv_white_balance_flat, hsv_image_flat)).reshape(-1)

            person_data.append(person_img_flatten)

        person_data = np.array(person_data)
        pca = PCA(n_components=2)
        person_data_reduce = pca.fit_transform(person_data)

        labels = self.kmeans.predict(person_data_reduce)
        centers = self.kmeans.cluster_centers_

        Team1color = list(map(int, self.cfg.GROUP.TEAM1.Color.split(',')))
        Team2color = list(map(int, self.cfg.GROUP.TEAM2.Color.split(',')))

        a_color, b_color = np.array(Team1color), np.array(Team2color)

        centers_reshaped = centers.reshape(2, 1, 2)
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






