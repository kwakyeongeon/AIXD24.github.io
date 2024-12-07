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
![1](https://github.com/user-attachments/assets/4f517007-24a3-4ad2-b00e-3efca4e26ff2)

<사진출처-현대자동차>
### **Motivation**
전기차는 전 세계적으로 빠르게 보급되고 있으나, 충전소 배치의 효율성은 여전히 전기차 사용자들의 요구를 충분히 충족시키지 못하고 있습니다.
충전소가 잘못 배치되면 사용자 접근성이 낮아지고, 충전 대기 시간이 길어지며, 지역 간 충전 인프라의 불균형 문제가 발생할 수 있습니다.
효율적인 충전소 배치는 전기차 사용자의 편의성을 높일 뿐만 아니라 에너지 자원을 효율적으로 활용하는 데도 기여할 수 있습니다.
이 프로젝트는 전기차 사용량 증가와 충전소 배치 문제를 데이터 기반으로 해결하려는 시도로, 다음을 목표로 합니다.

전기차 보급률 분석
지역별 전기차 분포 데이터와 기존 충전소 위치 데이터 수집 및 분석
이를 통해 전기차 사용자 경험을 개선하고, 충전소 배치의 효율성을 극대화할 수 있는 최적의 충전소 위치를 제안하고자 합니다.  

### **Objective**
- **1. 전기차 등록 대수와 충전소 분포 데이터를 결합하여 부족 지역 식별**  
  전기차 등록 대수와 현재 충전소 분포 데이터를 분석하여 충전소가 부족한 지역을 효과적으로 식별합니다. 이를 통해 충전소 배치의 불균형 문제를 구체적으로 파악할 수 있습니다.  

- **2. 딥러닝 모델을 활용한 최적 배치 방안 제안**  
  딥러닝 기술을 활용해 충전소 배치를 최적화할 수 있는 방안을 제안합니다. 모델은 지역 특성, 사용자 밀도, 교통 흐름 등 다양한 변수를 종합적으로 고려하여 데이터 기반의 예측 결과를 제공합니다.  

- **3. 데이터 시각화를 통한 설득력 있는 결과 제공**  
  분석 결과를 직관적으로 이해할 수 있도록 시각화 자료를 제작합니다. 이를 통해 사용자와 이해관계자들에게 설득력 있는 인사이트를 전달하며, 향후 충전소 인프라 개선을 위한 기초 자료로 활용합니다.  

---

## **II. Datasets**

### **Dataset Description**
1. **전기차 충전소 데이터**  
   - **출처**: [한국환경공단 공공데이터](https://www.data.go.kr/)  
   - **설명**: 설치년도, 위치, 충전 속도, 충전기 타입 등의 정보를 포함.  
   - **주요 변수**:  
     - `설치년도`, `시도`, `군구`, `위도/경도`, `충전기 타입`, `시설 구분`.

    ev_charging_station_data.csv (82176KB)

2. **전기차 등록 대수 데이터**  
   - **출처**: 공공데이터 API  
   - **설명**: 각 지역별 전기차 등록 대수와 연도별 증가 추이를 포함.  
   - **주요 변수**:  
     - `기준일`, `시도`, `전기차 대수`.

    ev_distribution_data.csv (2KB)

   ### **Data Preprocessing**
데이터는 다양한 전처리 과정을 거쳐 분석 가능한 형태로 가공되었습니다.

---

#### **Code 1: 시도 이름 표준화**
```python
# 데이터 전처리 함수: 시도 이름 표준화
def standardize_region_names(df, region_col):
    region_mapping = {
        "강원": "강원특별자치도",
        "경기": "경기도",
        "경남": "경상남도",
        "경북": "경상북도",
        "광주": "광주광역시",
        "대구": "대구광역시",
        "대전": "대전광역시",
        "부산": "부산광역시",
        "서울": "서울특별시",
        "세종": "세종특별자치시",
        "울산": "울산광역시",
        "인천": "인천광역시",
        "전남": "전라남도",
        "전북": "전북특별자치도",
        "제주": "제주특별자치도",
        "충남": "충청남도",
        "충북": "충청북도"
    }
    df[region_col] = df[region_col].replace(region_mapping)
    return df
```

#### **코드 설명**

**지역 이름 일관성 맞추기: 데이터 병합의 필수 단계**
충전소 데이터와 전기차 등록 대수 데이터를 결합하기 위해, region_mapping을 사용해 지역 이름을 표준화합니다. 이는 데이터 병합 과정에서 필수적인 단계로, 데이터 간 불일치로 인한 오류를 방지합니다.

**처리 단계**
**1. 지역 이름 매핑 테이블 생성**
서로 다른 데이터셋에 사용된 지역 이름의 변형(예: "서울특별시" vs "서울")을 확인하고, 이를 표준 이름으로 매핑하는 테이블을 작성합니다.

**2. 매핑 적용 및 표준화**
매핑 테이블을 각 데이터셋의 지역 이름 컬럼에 적용하여 일관성 있는 표준 이름으로 변환합니다.

**3. 데이터 병합**
표준화된 지역 이름을 기준으로 데이터를 병합하여 분석 가능한 통합 데이터셋을 생성합니다.

#### **Code 2: 데이터 병합**
```python

# 충전소 데이터 병합: 충전소 수 계산
charging_counts = charging_data.groupby("시도").size().reset_index(name="충전소 수")

# 전기차 등록 대수 데이터 변환 및 병합
ev_distribution_data = ev_distribution_data.melt(
    id_vars=["기준일"],
    var_name="시도",
    value_name="전기차 대수"
)
merged_data = ev_distribution_data.groupby("시도")["전기차 대수"].sum().reset_index()
merged_data = pd.merge(merged_data, charging_counts, on="시도", how="left").fillna(0)
```
#### **코드 설명**

**데이터 처리 및 병합 과정**


**1. 충전소 데이터에서 시도별 충전소 개수 계산**
충전소 데이터를 시도별로 그룹화하여 충전소 개수를 집계합니다. 이렇게 생성된 데이터는 분석의 기본 단위가 되는 시도별 충전소 분포를 나타냅니다. 이후 이를 데이터 프레임으로 변환하여 병합에 사용할 준비를 합니다.

**2. melt를 활용한 전기차 등록 데이터 변환**
전기차 등록 데이터는 다양한 컬럼으로 나뉘어 있을 수 있으므로, 이를 병합 가능한 구조로 변환하기 위해 melt를 사용합니다.
melt는 데이터를 "긴 형태"로 변환하여 시도를 기준으로 충전소 데이터와 쉽게 병합할 수 있게 합니다.
이 과정에서 "시도"와 "전기차 등록 대수"와 같은 키 컬럼을 생성합니다.

**3. 최종 데이터 병합**
충전소 데이터와 변환된 전기차 등록 데이터를 시도를 기준으로 병합하여, 각 시도의 충전소 개수와 전기차 등록 대수를 포함한 통합 데이터 프레임을 생성합니다.

 -이 데이터는 시도별 전기차 보급 상황과 충전 인프라의 격차를 분석하는 데 활용됩니다.
 -병합 이후 결측값은 적절히 처리(예: fillna(0))하여 완전한 데이터셋을 확보합니다.

#### **Code 3: 지역 이름 통합 및 충전소 부족률 계산**
```python

# 지역 이름 통합 함수
def merge_duplicate_regions(df):
    duplicate_mapping = {
        "전북특별자치도": "전라북도",
        "강원특별자치도": "강원특별자치도"
    }
    df["시도"] = df["시도"].replace(duplicate_mapping)
    df = df.groupby("시도", as_index=False).sum()
    return df

# 지역 이름 통합 및 데이터 재정렬
merged_data = merge_duplicate_regions(merged_data)

# 충전소 부족률 계산
merged_data["충전소 부족률"] = np.where(
    merged_data["충전소 수"] == 0, np.inf, merged_data["전기차 대수"] / merged_data["충전소 수"]
)
```
#### **코드 설명**

**데이터 정리 및 충전소 부족률 계산 과정**

**1. 지역 이름 중복 문제 해결**
데이터 병합 후 지역 이름이 중복으로 나타나는 경우, 동일한 지역을 그룹화하고 관련 데이터를 합산 또는 평균화하여 중복 문제를 해결합니다.

예: 동일 지역이 "서울"과 "서울특별시"로 중복되는 경우, region_mapping 또는 그룹화를 사용해 일관되게 정리합니다.


**2. 데이터 재정렬**
데이터를 지역(시도) 기준으로 정렬하여 결과를 직관적으로 분석할 수 있도록 재구성합니다.

정렬 기준: 시도명 또는 충전소 부족률, 필요에 따라 오름차순 또는 내림차순으로 정렬합니다.


**3. 충전소 부족률 계산**
충전소 부족률은 다음 공식으로 계산합니다:

충전소 부족률 = 전기차 대수/충전소 수

충전소가 없는 경우: 분모가 0이 되므로 부족률은 inf로 처리하여 별도로 표시하거나 분석에서 제외할 수 있도록 관리합니다.
계산된 부족률을 데이터 프레임에 새로운 컬럼으로 추가합니다.

**4. 결과물 구성**
최종 데이터는 아래와 같은 컬럼으로 구성됩니다:

시도: 지역 이름
전기차 대수: 해당 시도의 등록된 전기차 수
충전소 개수: 해당 시도의 충전소 개수
충전소 부족률: 전기차 대수 대비 충전소 부족 비율

#### **Code 4: 충전소 부족률 시각화**
```python

# 시각화
plt.figure(figsize=(12, 6))
plt.bar(merged_data["시도"], merged_data["충전소 부족률"], color='skyblue')
plt.xticks(rotation=45)
plt.ylabel("충전소 부족률")
plt.title("지역별 충전소 부족률")
plt.tight_layout()
plt.savefig("charging_station_deficit_updated.png")
plt.show()
```

![charging_station_deficit_updated](https://github.com/user-attachments/assets/db77d237-0025-4b1c-9f9f-d173668b9274)
#### **코드 설명**

Matplotlib을 사용해 시각화한 충전소 부족률 그래프는 지역별 전기차와 충전소 인프라의 불균형을 한눈에 보여줍니다. 
그래프를 통해 부족률이 높은 지역을 쉽게 파악할 수 있어, 우선적으로 충전소가 필요한 곳을 명확히 확인할 수 있습니다. 
또한, 충전소와 전기차의 균형이 잘 맞는 지역이나 충전소가 전혀 없는 지역도 직관적으로 비교할 수 있습니다. 
이런 시각화는 분석 결과를 이해하기 쉽게 만들어주며, 정책 수립이나 투자 판단에 실질적인 도움을 줄 수 있습니다.

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
- 딥러닝 관련 논문 및 참고 자료.  


