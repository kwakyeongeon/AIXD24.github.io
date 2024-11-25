# **Optimizing EV Charging Station Placement with Deep Learning**

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
    - [Evaluation Metrics](#evaluation-metrics)  
    - [Visualization](#visualization)  
6. [V. Related Work](#v-related-work)  
7. [VI. Conclusion](#vi-conclusion)  

---

## **Members**
- 곽연건, 한양대학교 서울캠퍼스 기계공학부, kwak6072@hanyang.ac.kr  
- 정우진, 한양대학교 서울캠퍼스 기계공학부  

---

## **I. Proposal**

### **Motivation**
- 왜 전기차 충전소 최적 배치에 대한 연구를 진행하는가?  
  - 전기차 사용 증가에 따라 충전소 부족 문제를 해결하기 위해.  
  - 지역별 충전소 수요를 데이터 기반으로 예측하여 효과적 배치 방안을 도출하기 위해.  
  - 지속 가능한 교통 인프라를 지원하고, 전기차 사용을 더욱 촉진하기 위해.  

### **Objective**
- 데이터 기반으로 분석한 충전소 위치의 최적 배치 제안.  
- 지역별 전기차 보급률 및 기존 충전소 밀도를 고려한 배치 권장 사항 제공.  

---

## **II. Datasets**

### **Dataset Description**
1. **전기차 충전소 데이터**  
   - **출처**: [한국환경공단 공공데이터](https://www.data.go.kr/)  
   - **설명**: 설치년도, 위치, 충전 속도, 충전기 타입 등의 정보를 포함.  
   - **주요 변수**:  
     - `설치년도`, `시도`, `군구`, `위도/경도`, `충전기 타입`, `시설 구분`.  

2. **전기차 등록 대수 데이터**  
   - **출처**: 공공데이터 API  
   - **설명**: 각 지역별 전기차 등록 대수와 연도별 증가 추이를 포함.  
   - **주요 변수**:  
     - `기준일`, `시도`, `전기차 대수`.  

---

## **III. Methodology**

### **Algorithms and Models**
- **지도 학습 기반 딥러닝 모델**:  
  - **LSTM**: 시간에 따른 전기차 등록 대수의 증가 추이 예측.  
  - **CNN**: 지리적 패턴 분석 및 충전소의 최적 위치 선정.  
- **예측 모델 구성 및 훈련 과정**:  
  - 기존 데이터셋에서 충전소가 부족한 지역 탐지.  
  - 새로운 충전소 위치를 예측하기 위한 회귀 분석 모델 훈련.  

### **Features**
- 주요 피처:  
  - 지역별 전기차 대수.  
  - 기존 충전소와의 거리.  
  - 충전소 밀도 및 유형.  
  - 위도/경도.  

---

## **IV. Evaluation & Analysis**

### **Evaluation Metrics**
- **RMSE (Root Mean Square Error)**: 위치 예측 정확도 측정.  
- **R² (결정계수)**: 모델의 설명력을 평가.  

### **Visualization**
- **히트맵**:  
  - 기존 충전소 밀도를 시각적으로 표현하여 충전소가 부족한 지역을 식별.  
- **시계열 그래프**:  
  - 전기차 대수의 시간적 증가 추이를 보여줌.  

---

## **V. Related Work**

### **참조한 문헌 및 도구**
- **도구**: Python (`pandas`, `numpy`, `TensorFlow`, `matplotlib`), Google Colab.  
- **관련 연구 및 블로그**:  
  - [Kaggle의 EV 충전소 데이터 분석 예제](https://www.kaggle.com/)  
  - [논문: 전기차 인프라와 충전소 최적 배치](https://scholar.google.com/)  

---

## **VI. Conclusion**

### **Discussion**
- 충전소 최적 배치의 경제적, 환경적 이점 논의.  
- 데이터 기반 인프라 계획의 가능성을 보여줌.  
- 연구의 한계:  
  - 추가적인 충전소 사용 패턴 데이터 필요.  
  - 설치 비용, 사용자 편의성 같은 비정형 데이터 활용 필요.  

### **Future Work**
- 충전소 유형별 최적 배치 연구.  
- 사용자 이동 데이터를 추가하여 실시간 충전소 추천 시스템 구축.  

---

## **References**
- [한국환경공단 공공데이터](https://www.data.go.kr/)  
- [Kaggle: EV 충전소 데이터](https://www.kaggle.com/)  
- 딥러닝 관련 논문 및 참고 자료.  


