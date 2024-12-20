# **Optimizing EV Charging Station Placement with Deep Learning**

**유튜브영상링크**

[2024_2학기_AIX딥러닝기말프로젝트][https://www.youtube.com/watch?v=qrGMGiVBvos]

---

## **Contents**
1. [Members](#members)  
2. [I. Proposal](#i-proposal)  
    - [Motivation](#motivation)  
    - [Objective](#Objective)  
3. [II. Datasets](#ii-datasets)  
    - [Dataset Description](#dataset-description)  
4. [III. Methodology](#iii-methodology)  
    - [Algorithms and Models](#algorithms-and-models)  
    - [Features](#features)  
5. [IV. Evaluation & Analysis](#iv-evaluation--analysis)
    - [Evaluation Metrics](#Evaluation-Metrics)
    - [Visualization](#Visualization)
    - [Summary of Findings](#Summary-of-Findings)   
7. [V. Related Work](#v-related-work)  
8. [VI. Conclusion](#vi-conclusion)  

---

## **Members**
- 곽연건, 한양대학교 서울캠퍼스 기계공학부, kwak6072@hanyang.ac.kr, 총괄

- 정우진, 한양대학교 서울캠퍼스 기계공학부, 이메일, 보조

---

## **I. Proposal**
![1](https://github.com/user-attachments/assets/4f517007-24a3-4ad2-b00e-3efca4e26ff2)

<사진출처-현대자동차>
### **Motivation**
 전기차는 전 세계적으로 빠르게 보급되고 있으나, 충전소 배치의 효율성은 여전히 전기차 사용자들의 요구를 충분히 충족시키지 못하고 있습니다.
 
충전소가 잘못 배치되면 사용자 접근성이 낮아지고, 충전 대기 시간이 길어지며, 지역 간 충전 인프라의 불균형 문제가 발생할 수 있습니다.

효율적인 충전소 배치는 전기차 사용자의 편의성을 높일 뿐만 아니라 에너지 자원을 효율적으로 활용하는 데도 기여할 수 있습니다.

이 프로젝트는 전기차 사용량 증가와 충전소 배치 문제를 데이터 기반으로 해결하려는 시도로, 다음을 목표로 합니다.

**전기차 보급률 분석**
지역별 전기차 분포 데이터와 기존 충전소 위치 데이터 수집 및 분석

이를 통해 전기차 사용자 경험을 개선하고, 충전소 배치의 효율성을 극대화할 수 있는 최적의 충전소 위치를 제안하고자 합니다.  

### **Objective**
- **1. 전기차 등록 대수와 충전소 분포 데이터를 결합하여 부족 지역 식별**  
  전기차 등록 대수와 현재 충전소 분포 데이터를 분석하여 충전소가 부족한 지역을 효과적으로 식별합니다.
  
  이를 통해 충전소 배치의 불균형 문제를 구체적으로 파악할 수 있습니다.  

- **2. 딥러닝 모델을 활용한 최적 배치 방안 제안**  
  딥러닝 기술을 활용해 충전소 배치를 최적화할 수 있는 방안을 제안합니다. 모델은 지역 특성, 사용자 밀도,
  
  교통 흐름 등 다양한 변수를 종합적으로 고려하여 데이터 기반의 예측 결과를 제공합니다.  

- **3. 데이터 시각화를 통한 설득력 있는 결과 제공**  
  분석 결과를 직관적으로 이해할 수 있도록 시각화 자료를 제작합니다. 이를 통해 사용자와 이해관계자들에게 설득력 있는 인사이트를 전달하며,
  
  향후 충전소 인프라 개선을 위한 기초 자료로 활용합니다.  

---

## **II. Datasets**

### **Dataset Description**
1. **전기차 충전소 데이터**  
   - **출처**: [한국환경공단 공공데이터](https://www.data.go.kr/)  
   - **설명**: 설치년도, 위치, 충전 속도, 충전기 타입 등의 정보를 포함.  
   - **주요 변수**:  
     - `설치년도`, `시도`, `군구`, `위도/경도`, `충전기 타입`, `시설 구분`.

    ev_charging_station_data.csv (82176KB)

   ### **Data Preprocessing**
   데이터는 다양한 전처리 과정을 거쳐 분석 가능한 형태로 가공되었습니다.

---

#### **Code 1: 데이터 로드**
```python
import pandas as pd

def load_charging_station_data(data_path):
    """
    충전소 데이터를 로드합니다.
    """
    return pd.read_csv(data_path, header=None).dropna()

```

#### **코드 설명**

load_charging_station_data 함수는 충전소 데이터를 CSV 파일에서 불러옵니다.

데이터에 결측값이 존재하면 이를 제거하여 데이터의 완전성을 유지합니다.




#### **Code 2: 데이터 전처리 및 스케일링**
```python

from sklearn.preprocessing import MinMaxScaler

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

```
#### **코드 설명**

data_preprocess 함수는 충전소 데이터에서 위도와 경도를 추출하여 전처리합니다.

MinMaxScaler를 사용하여 데이터를 0과 1 사이로 스케일링하여 학습 안정성을 높입니다.

데이터는 80%를 학습용, 20%를 테스트용으로 나누어 모델 학습과 평가를 분리합니다.

---

#### **결론**

데이터셋 전처리는 본 프로젝트의 핵심 단계로, 다음과 같은 작업을 수헹.

충전소 데이터와 전기차 등록 데이터를 로드 및 전처리.

위도와 경도 정보를 추출하여 스케일링 후 학습용/테스트용으로 분리.

데이터의 지리적 분포를 시각화하여 추가 분석과 모델 학습에 필요한 인사이트를 도출.

이를 통해 전기차 충전소 부족 문제를 해결할 기반 데이터를 성공적으로 구축.


---

## **III. Methodology**

### **Algorithms and Models**

본 프로젝트에서는 전기차 충전소 최적 배치를 위해 Autoencoder를 기반으로 한 딥러닝 모델을 사용하였습니다. 

Autoencoder는 입력 데이터를 잠재 공간(latent space)으로 압축한 뒤, 

이를 복원하는 과정을 통해 데이터의 특징을 학습하는 비지도 학습 모델입니다. 

이 모델은 다음과 같은 두 가지 주요 컴포넌트로 구성됩니다:

**1.Encoder**
Encoder는 입력 데이터를 저차원 잠재 공간(latent space)으로 압축합니다. 

이 과정에서 데이터의 핵심적인 특징만 유지됩니다.

**2. Decoder**
Decoder는 잠재 공간의 표현(latent representation)을 기반으로 원래 입력 데이터를 복원합니다.

이를 통해 모델은 입력 데이터를 압축하고 복원하는 과정을 반복 학습하여 데이터의 구조를 학습합니다.

**Model Architecture**

**Code Implementation**

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    """
    Autoencoder 모델 정의
    """
    # Encoder
    latent_dim = 2
    encoder_input = layers.Input(shape=(2,))
    encoded = layers.Dense(64, activation='relu')(encoder_input)
    latent = layers.Dense(latent_dim, activation='linear')(encoded)
    encoder = models.Model(encoder_input, latent, name="encoder")

    # Decoder
    decoder_input = layers.Input(shape=(latent_dim,))
    decoded = layers.Dense(64, activation='relu')(decoder_input)
    output = layers.Dense(2, activation='linear')(decoded)
    decoder = models.Model(decoder_input, output, name="decoder")

    # Autoencoder
    autoencoder_input = layers.Input(shape=(2,))
    encoded_latent = encoder(autoencoder_input)
    decoded_output = decoder(encoded_latent)
    autoencoder = models.Model(autoencoder_input, decoded_output, name="autoencoder")

    # 모델 컴파일
    autoencoder.compile(optimizer='adam', loss='mse', metrics=["mae"])
    return autoencoder
```

**How does an Autoencoder work?**

Autoencoder는 입력값을 재구성하여 출력합니다. 

Autoencoder는 인코더와 디코더라는 2개의 더 작은 신경망으로 구성됩니다. 

훈련 중에 인코더는 입력 데이터로부터 잠재 표현이라고 하는 일련의 특징을 학습합니다. 

이와 동시에 디코더는 그러한 특징을 토대로 데이터를 재구성하도록 훈련됩니다. 

그러면 autoencoder는 이전에 본 적이 없는 입력값을 예측하는 데 적용할 수 있습니다. 

Autoencoder는 일반화가 매우 용이하며 영상, 시계열, 텍스트 등 다양한 데이터형에 사용할 수 있습니다.


오토인코더 신경망의 아키텍처. 인코더는 입력의 잠재 표현을 생성합니다. 
이후, 해당 표현이 디코더로 입력됩니다.

![1](https://github.com/user-attachments/assets/c149b760-4769-4c21-9133-41262dac322b) 

[그림1]: Autoencoder는 인코더와 디코더로 구성됩니다.

---

**Applications of Autoencoders**

Autoencoder는 인코더가 훈련될 때 입력값의 모든 잡음을 자연적으로 무시합니다. 

이 기능은 입력값과 출력값이 비교될 때 잡음을 제거하거나 이상을 감지하는 데 이상적입니다. 
(그림 2 및 3 참조)

오토인코더는 영상(빨간색 r이 있는 점이 찍힌 배경)에서 잡음(빨간색 r)을 제거합니다.
![2](https://github.com/user-attachments/assets/b7ba0074-0559-4634-aa5f-efd92c128b77) 

[그림2]: 영상에서 잡음 제거.

---

오토인코더는 영상(빨간색 r이 있는 점이 찍힌 배경)에서 이상(빨간색 r)을 검출합니다.

![2-2](https://github.com/user-attachments/assets/d062eb09-e208-4551-984a-c9eafcccb5b5)

[그림3]: 영상 기반 이상 감지.

---

잠재 표현은 합성 데이터 생성에 사용할 수도 있습니다. 예를 들면 실제 같아 보이는 손글씨나 텍스트 문구를 자동으로 작성할 수 있습니다. (그림 4)

오토인코더의 입력 텍스트로 셰익스피어의 소네트를 사용했습니다. 

출력 텍스트는 생성된 소네트입니다.

![3](https://github.com/user-attachments/assets/8014f4e7-23a0-4f23-b861-c6b22a724286) 

[그림4]: 기존 텍스트로부터 새 텍스트 문구 생성하기.

---

시계열 기반 autoencoder는 신호 데이터의 이상을 감지하는 데 사용할 수도 있습니다. 

예를 들어 예측 정비에서는 산업 기계에서 수집한 정상 동작 데이터로 autoencoder를 훈련시킬 수 있습니다. (그림 5)

오토인코더는 산업 기계의 정상 동작 데이터(시계열 신호)에서 오차를 검출하여 제거합니다.

![4](https://github.com/user-attachments/assets/1d7052a6-d820-46f6-8480-9e96af66798d) 

[그림5]: 예측 정비를 위해 정상 동작 데이터로 훈련시키기.

---

이렇게 훈련된 autoencoder는 이후에 새로운 수신 데이터로 테스트를 거칩니다. 

Autoencoder 출력값으로부터의 변동이 크면 이상 동작임을 가리키는 것이며, 이렇게 되면 조사가 필요할 수 있습니다. (그림 6)

오토인코더는 산업 기계의 비정상 동작 데이터(시계열 신호)에서 큰 오류를 검출하여 제거합니다.

![5](https://github.com/user-attachments/assets/00b7853b-7813-4842-98fc-f57939404b33) 

[그림 6]: 입력 데이터의 이상을 표시하는 커다란 오차. (정비가 필요하다는 신호일 수 있음)

---

#### **Why Autoencoder?**

**Autoencoder는 다음과 같은 이유로 본 프로젝트에 적합합니다:**

-공간 데이터의 학습: Autoencoder는 전기차 충전소의 위도와 경도 데이터를 학습하여 주요 패턴을 압축적으로 표현할 수 있습니다.

-결손 데이터 예측: Autoencoder는 학습된 패턴을 기반으로 부족한 충전소 위치를 복원하거나 예측하는 데 유용합니다.

-군집화와 시각화: Latent space를 기반으로 충전소 위치를 군집화하고, 이를 분석하여 최적의 위치를 제안할 수 있습니다.



**Training Process**

**Data Preprocessing:**

-데이터는 위도와 경도 값으로 구성되며, 이를 MinMaxScaler를 사용하여 스케일링.

학습 데이터와 테스트 데이터를 80:20 비율로 분리.

-mean squared error (MSE)와 mean absolute error (MAE)를 손실 함수로 사용하여 학습.

100 epochs 동안 학습하며, 학습 및 검증 손실 추이를 시각화.

-Autoencoder의 복원 결과를 테스트 데이터에 대해 평가.

-Latent space를 기반으로 새로운 충전소 위치를 추천.

---
#### **modules.py**
 ```python

import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():

    # Encoder 모델
    latent_dim = 2  # 잠재 공간의 차원
    encoder_input = layers.Input(shape=(2,))
    encoded = layers.Dense(64, activation='relu')(encoder_input)
    latent = layers.Dense(latent_dim, activation='linear')(encoded)

    encoder = models.Model(encoder_input, latent, name="encoder")

    # Decoder 모델
    decoder_input = layers.Input(shape=(latent_dim,))
    decoded = layers.Dense(64, activation='relu')(decoder_input)
    output = layers.Dense(2, activation='linear')(decoded)

    decoder = models.Model(decoder_input, output, name="decoder")

    # Autoencoder 모델
    autoencoder_input = layers.Input(shape=(2,))
    encoded_latent = encoder(autoencoder_input)
    decoded_output = decoder(encoded_latent)

    autoencoder = models.Model(autoencoder_input, decoded_output, name="autoencoder")

    # 모델 컴파일
    autoencoder.compile(optimizer='adam', loss='mse', metrics=["mae"])

    return autoencoder

```
**코드 설명**

**Model Overview**

**Encoder:**

입력 데이터를 2차원 잠재 공간으로 압축.

주요 레이어: Dense(64, ReLU), Dense(2, Linear).

**Decoder:**

잠재 공간 데이터를 원래 형태로 복원.

주요 레이어: Dense(64, ReLU), Dense(2, Linear).

**Autoencoder:**

Encoder와 Decoder를 결합하여 입력 데이터의 패턴을 학습.

Optimization

Optimizer: Adam

Loss Function: Mean Squared Error (MSE)

Metrics: Mean Absolute Error (MAE)

---

#### **utils.py**

 ```python
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

```

**코드 설명**

**1. 데이터 전처리**

데이터 로드 및 변환: CSV 파일에서 충전소 위치 데이터를 읽고, 위도와 경도를 분리하여 numpy 배열로 변환.

스케일링: MinMaxScaler로 데이터를 정규화하여 학습 안정성과 정확도를 향상.

데이터 분할: 학습과 테스트 데이터로 나누어 모델 훈련과 성능 평가에 활용.

**2. 모델 학습**

**Autoencoder 모델 설계**

Encoder: 데이터를 압축해 핵심 특징 학습.

Decoder: 데이터를 복원하여 입력과 유사하게 재구성.

**학습 과정**

손실 기준: MSE, 보조 지표: MAE.

과적합 방지를 위해 학습 데이터와 검증 데이터 활용.

**3. 평가 및 시각화**

**평가 지표**

MSE: 평균 제곱 오차로 예측 정확성을 평가.

MAE: 평균 절대 오차로 모델 성능 평가.

**시각화**

학습 손실 변화 그래프와 기존 위치(초록색)와 추천 위치(빨간색) 비교 그래프.

**4. 데이터 군집화 및 위치 보정**

**KMeans 클러스터링**

최적 충전소 위치를 클러스터링으로 식별.

각 클러스터 중심과 외곽선을 시각화.

**위치 보정**

모델 추천 위치를 데이터 분포에 따라 조정해 현실성을 강화.

---


#### **main.py**

 ```python

import numpy as np
from modules.modules import build_model
from modules.utils import data_preprocess, plot_history, plot_results, cluster, adjust

if __name__ == '__main__':

    data_path = './data/ev_charging_station_data.csv'

    # 데이터 전처리    
    train_x, test_x = data_preprocess(data_path)

    # 모델 빌드 및 학습
    model = build_model()
    history = model.fit(train_x, train_x, epochs=100, batch_size=32, validation_data=(test_x, test_x))
    
    # 모델 결과 출력
    plot_history(history)
    plot_results(train_x, test_x, model)

    # 데이터셋 군집화 
    cluster(train_x, test_x, model)
    adjust(test_x, model)

    # 랜덤한 더미 데이터 삽입 및 위치 보정 

    # 데이터의 분포를 통해 사람이 전기차 충전소의 대략적인 위치를 입력해주면 기존 데이터의 분포를 고려하여 적절한 위치로 보정해줌 
    

```
**코드 설명**

<메인 파일 참조>

### **Features**

- 주요 피처:  

- 충전소 밀도 및 유형.  

- 위도/경도.  

---

## **IV. Evaluation & Analysis**

**Evaluation Metrics and Visualization**

### **Evaluation Metrics**
**Mean Squared Error (MSE)**

정의: MSE는 예측 값과 실제 값 간의 차이를 제곱한 후 평균을 계산합니다. 이는 예측이 얼마나 정확한지를 수치적으로 표현합니다.

활용 이유: MSE는 큰 오차에 민감하게 반응하여 모델의 성능을 세밀하게 평가합니다. 작은 값일수록 모델의 예측이 실제 값과 가깝다는 것을 의미합니다.

실험에서의 사용: 학습 데이터와 검증 데이터에 대해 MSE 값을 계산하여 학습 진행 상태를 모니터링했습니다.

**Mean Absolute Error (MAE)**

정의: MAE는 예측 값과 실제 값 간의 절대 오차의 평균입니다. 이는 오차의 방향성을 제거하고 평균적인 차이를 수치적으로 표현합니다.

활용 이유: MAE는 데이터에 대한 오차의 전반적인 경향을 파악하는 데 유용합니다. MSE보다 이상치(outlier)에 덜 민감합니다.

실험에서의 사용: 학습 과정 중 데이터의 일반적인 오차를 평가하여 모델의 안정성을 검토했습니다.

---

### **Visualization**
**Training and Validation Loss Graphs**

MSE 손실 그래프: 학습 데이터와 검증 데이터의 MSE 값이 Epoch에 따라 어떻게 변화하는지 나타냅니다. 

학습과 검증 손실이 감소하여 모델이 데이터를 적절히 학습하고 있음을 확인했습니다.

![MSE](https://github.com/user-attachments/assets/e7190e73-5628-4867-85df-918bed548566)

[MSE 손실 그래프]

MAE 손실 그래프: Epoch에 따른 MAE 값의 변화를 시각적으로 표현하여 학습 데이터와 검증 데이터의 평균 오차 추이를 확인했습니다.

---

![MAE](https://github.com/user-attachments/assets/bb48b5b6-dd21-4b8f-87e4-b3dd68907202)

[MAE 손실 그래프]

---

**랜덤 데이터 실험 결과 플롯**

실험 결과, 랜덤으로 선택된 데이터 위치를 모델이 학습 데이터의 분포를 따르도록 보정한 결과를 시각적으로 확인했습니다.

-초록색 점: 학습 데이터에서의 기존 충전소 위치.

-붉은색 점: 모델이 추천한 조정된 충전소 위치.

![plot_res](https://github.com/user-attachments/assets/0d4a1bb1-8a0a-4564-9549-42f718560dc9)

[학습 데이터 내 충전소 위치 VS. 모델을 통해 보정된 임의의 위치 데이터]

---

**학습된 오토인코더의 잠재벡터 클러스터링**

-오토인코더의 잠재 벡터를 클러스터링 한 뒤, 원본 이미지 영역에서 이 클러스터를 보여줌으로 써 적절한 분포가 학습되었는가를 평가합니다.

-인/부천 및 서울 서부권의 클러스터가 동일한 클러스터로 판단된 것으로 보아, 충전소의 위치가 인/부천 및 서울권에 집중되었음을 확인할 수 있습니다. 

![cluster](https://github.com/user-attachments/assets/383183f4-4fb4-4ad6-826c-396845cb6f2d)

[학습된 오토인코더의 잠재벡터 클러스터링]

---

**평가 데이터 및 오차 플롯**

-평가 데이터의 원본 (위도, 경도)와 모델의 출력 (위도, 경도) 데이터를 플롯함으로 써, 

학습 모델이 얼만큼 데이터의 분포를 내포하고 있는가에 대한 간단한 비교가 가능합니다.

-초록 점은 평가 데이터이며, 빨간 점은 조정된 데이터를 나타내고, 두 데이터 간의 오차가 거의 없음을 확인할 수 있습니다.

![adjust](https://github.com/user-attachments/assets/ee0c67c0-1b72-483f-9ccd-4331db8dbd7b)

[오차플롯]

---

### **Summary of Findings**

-모델은 기존 충전소 데이터의 분포를 학습하고 이를 기반으로 부족한 지역에 새로운 충전소를 추천할 수 있었습니다.

-MSE와 MAE 값이 낮아 학습 데이터의 분포를 정확히 내포하였고, 검증 데이터에서도 일관된 성능을 보였습니다.

-시각화를 통해 모델의 학습 과정과 결과를 직관적으로 이해할 수 있었으며, 
클러스터링 및 보정된 데이터 결과는 실질적인 충전소 위치 추천에 활용 가능성이 높음을 시사했습니다.

---

## **V. Related Work**

### **참조한 문헌 및 도구**
- **도구**: Python (`pandas`, `numpy`, `TensorFlow`, `matplotlib`), Google Colab.  

- **관련 연구 및 블로그**:

  -[한국환경공단 공공데이터](https://www.data.go.kr/)
    
  -[TensorFlow](https://kr.mathworks.com/discovery/autoencoder.html)
  
  -[MathWorks](https://www.tensorflow.org/tutorials/generative/autoencoder?hl=ko) 


---

## **VI. Conclusion**

### **Discussion**

본 연구는 전기차 충전소의 최적 배치가 가져올 경제적, 환경적 이점을 강조했습니다. 

효율적인 배치를 통해 충전 인프라의 활용도를 높이고, 불필요한 에너지 소비를 줄여 환경 보호에 기여할 수 있음을 확인했습니다.

데이터 기반 접근 방식이 인프라 계획 수립에 있어 높은 잠재력을 가진다는 점도 보여주었습니다. 

특히, 전기차 보급률과 충전소 간의 불균형 문제를 정량적으로 분석할 수 있는 틀을 제공했습니다.

하지만 몇 가지 한계점도 존재합니다. 충전소 사용 패턴에 대한 상세 데이터가 부족하여, 사용자의 실제 충전 습관을 반영하지 못했다는 점이 가장 큰 제약입니다. 

또한, 설치 비용, 사용자 편의성, 지역 정책 등 비정형 데이터를 반영하지 못한 점도 개선해야 할 부분으로 지적됩니다.


### **Future Work**

**1. 충전소 유형별 최적 배치 연구**

-급속 충전소와 완속 충전소의 서로 다른 특성을 고려한 최적 배치 전략을 개발할 필요가 있습니다. 이를 통해 다양한 사용자 요구를 충족할 수 있는 세분화된 접근법을 마련할 수 있습니다.

**2. 사용자 이동 데이터를 활용한 실시간 충전소 추천 시스템**

-전기차 운전자의 이동 경로 데이터를 추가하여, 실시간으로 가장 적합한 충전소를 추천하는 시스템을 구축할 계획입니다. 이는 충전소의 효율성을 높이고 사용자 편의를 극대화할 수 있는 중요한 발전 방향입니다.

**3. 기름 주유소 데이터 통합을 통한 최적 위치 탐색**

-기존의 기름 주유소 위치 데이터를 포함시켜, 현재의 교통 및 인프라 네트워크와 통합된 최적의 전기차 충전소 위치를 탐색하는 연구를 진행할 예정입니다. 이는 기존 인프라를 활용해 경제성과 실현 가능성을 더욱 높이는 데 기여할 것입니다.

**4. 비정형 데이터 활용 확대**

-사용자 설문조사, 지역별 정책 데이터, 설치 비용과 같은 비정형 데이터를 반영하여, 더욱 정교하고 현실적인 충전소 배치 모델을 설계할 계획입니다.



---

