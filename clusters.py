import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


data = []
n_clusters = 3
for i in range(n_clusters):
    filename = glob.glob(f"./kmeans_fit_data(R&Y&B)/{i}/*.jpg")
    # cv2.imwrite(f"check{i}.png", cv2.resize(cv2.imread(filename[0]), (20, 40), interpolation=cv2.INTER_AREA))
    data += [cv2.resize(cv2.imread(name), (20, 40), interpolation=cv2.INTER_AREA).reshape(-1) for name in filename]


X = np.array(data)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
for cluster in range(n_clusters):
    cluster_points = X_reduced[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster + 1}')

for i in range(X_reduced.shape[0]):
    plt.text(X_reduced[i, 0], X_reduced[i, 1], str(i), fontsize=9, ha='right')

# 添加標籤和標題
plt.title("KMeans Clustering of Images")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.legend()
plt.grid(True)
plt.savefig('RvsYvsB.png')