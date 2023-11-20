import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import MaxNLocator

data = pd.read_csv(r'D:\Kenkyu\Research\DATA\MHW\Train\NMHW.csv')

# 1. 孤立森林
clf1 = IsolationForest(contamination=0.1)
outliers1 = clf1.fit_predict(data)

# 2. One-Class SVM
clf2 = OneClassSVM()
outliers2 = clf2.fit_predict(data)

# 3. DBSCAN
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
clf3 = DBSCAN(eps=0.5, min_samples=5)
outliers3 = clf3.fit_predict(data_scaled)

# 4. LOF
clf4 = LocalOutlierFactor()
outliers4 = clf4.fit_predict(data)

# 5. K均值
clf5 = KMeans(n_clusters=2, n_init=10)
outliers5 = clf5.fit_predict(data)

# 6. 高斯混合模型
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
clf6 = GaussianMixture(n_components=2)
outliers6 = clf6.fit_predict(data_scaled)

# 7. 自编码器
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
clf7 = PCA(n_components=1)
data_pca = clf7.fit_transform(data_scaled)
reconstructed_data = clf7.inverse_transform(data_pca)
reconstruction_error = np.mean((data_scaled - reconstructed_data) ** 2, axis=1)
threshold = np.percentile(reconstruction_error, 95)
outliers7 = (reconstruction_error < threshold).astype(int)

# 8. 随机投影
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
clf8 = PCA(n_components=1, random_state=0)
data_pca = clf8.fit_transform(data_scaled)
outliers8 = ((data_scaled - clf8.inverse_transform(data_pca)) ** 2).sum(axis=1)

outliers1 = np.where(outliers1 == -1, 1, 0)
outliers2 = np.where(outliers2 == -1, 1, 0)
outliers3 = np.where(outliers3 == -1, 1, 0)
outliers4 = np.where(outliers4 == -1, 1, 0)
outliers5 = np.where(outliers5 == 1, 0, 1)
outliers6 = np.where(outliers6 == 0, 1, 0)
outliers7 = np.where(outliers7 == 1, 0, 1)
outliers8 = np.where(outliers8 > np.percentile(outliers8, 95), 1, 0)

# 计算每个outliers中0和1的数量，并将数量多的分组修改为0，数量少的分组修改为1
def adjust_outliers(outliers):
    count_0 = np.sum(outliers == 0)
    count_1 = np.sum(outliers == 1)
    if count_0 < count_1:
        temp = outliers.copy()
        outliers[outliers == 0] = 1
        outliers[temp == 1] = 0
    return outliers

outliers1 = adjust_outliers(outliers1)
outliers2 = adjust_outliers(outliers2)
outliers3 = adjust_outliers(outliers3)
outliers4 = adjust_outliers(outliers4)
outliers5 = adjust_outliers(outliers5)
outliers6 = adjust_outliers(outliers6)
outlies7 = adjust_outliers(outliers7)

outliers8 = adjust_outliers(outliers8)

outliers = np.stack([outliers1, outliers2, outliers3, outliers4, outliers5, outliers6, outliers7, outliers8])
outliers = outliers.T
methods = ["Isolation Forest", "One-Class SVM", "DBSCAN", "LOF", "K Means", "Gaussian Mixture Model", "Autoencoder", "Random Projection"]
#outliers_list = [outliers1.T, outliers2.T, outliers3.T, outliers4.T, outliers5.T, outliers6.T, outliers7.T, outliers8.T]
outlier = pd.DataFrame(outliers)
outlier.to_csv(r'D:\Kenkyu\Research\DATA\MHW\Outliers\Outliers_MHW.csv', index=False, header=methods)

"""""
plt.figure(figsize=(15, 10))
for i, (method, outliers) in enumerate(zip(methods, outliers_list), 1):
    ax = plt.subplot(4, 2, i)
    ax.plot(range(len(data)), data, color='#ccc', linestyle='-', zorder=1)
    ax.scatter(np.where(outliers == 0), data[outliers == 0], c='blue', label='0', marker='o', s=20, zorder=2)
    ax.scatter(np.where(outliers == 1), data[outliers == 1], c='red', label='1', marker='o', s=40, zorder=2)

# 设置水平轴的刻度定位器为整数
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(method)

plt.tight_layout()
plt.show()
"""""
