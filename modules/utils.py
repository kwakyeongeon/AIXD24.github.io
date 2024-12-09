import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

from tqdm import tqdm

from scipy.spatial import ConvexHull

def data_preprocess(
        data_path:str,
        train_ratio:float=0.8
):
    data = pd.read_csv(data_path, header=None).dropna()
    location = data.iloc[1:].to_numpy()

    # 좌표 추출
    inputs = []
    for data in tqdm(location):
        x = float(data[-1].split(',')[0])
        y = float(data[-1].split(',')[1])
        inputs.append([x, y])
    inputs = np.array(inputs)
    
    # 데이터 스케일링
    scaler = MinMaxScaler()
    inputs = scaler.fit_transform(inputs)

    # 학습 평가 분할
    train_x = inputs[:int(inputs.shape[0]*train_ratio), :]
    test_x = inputs[int(inputs.shape[0]*train_ratio):, :]

    return train_x, test_x


def plot_history(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # plot history
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training MSE')
    plt.plot(val_loss, label='Validation MSE', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('./results/MSE.png')
    plt.clf()

    train_loss = history.history['mae']
    val_loss = history.history['val_mae']
    
    # plot history
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training MAE')
    plt.plot(val_loss, label='Validation MAE', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('./results/MAE.png')
    plt.clf()

def plot_results(train_x, test_x, model):

    plt.scatter(train_x[:, 1], train_x[:, 0], color='green', s=1, label='exists')
    
    predictions = model.predict(test_x)
    plt.scatter(predictions[:, 1], predictions[:, 0], color='red', s=1, label='recommemd')
    plt.legend()
    plt.savefig('./results/plot_res.png')
    plt.clf()

def cluster(train_x, test_x, model):

    # KMeans 클러스터링
    n_clusters = 16
    machine = KMeans(n_clusters=n_clusters)
    machine.fit(train_x)

    labels = machine.labels_
    centers = machine.cluster_centers_

    # 시각화
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap("tab10", n_clusters)  # 클러스터 색상

    for i in range(n_clusters):
        # 클러스터 데이터 추출
        cluster_points = train_x[labels == i]
        
        # 데이터 점 시각화
        plt.scatter(cluster_points[:, 1], cluster_points[:, 0], label=f"Cluster {i + 1}", color=colors(i))
        
        # 클러스터 최외곽선 (Convex Hull)
        if len(cluster_points) >= 3:  # ConvexHull은 최소 3개의 점이 필요
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                plt.plot(cluster_points[simplex, 1], cluster_points[simplex, 0], color=colors(i))

    # 클러스터 중심 시각화
    plt.scatter(centers[:, 1], centers[:, 0], c='red', marker='X', s=200, label='Centroids')

    # 그래프 설정
    plt.title('KMeans Clustering with Convex Hulls')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/cluster.png')
    plt.clf()

def adjust(test_x, model):

    test_x = test_x[:30]

    adjusted = model(test_x)

    plt.scatter(test_x[:,1], test_x[:,0], s=5, color='g', label='inputs')
    plt.scatter(adjusted[:,1], adjusted[:,0], s=5, color='r', label='adjusted')
    plt.legend()
    plt.savefig('./results/adjust.png')
    plt.clf()
