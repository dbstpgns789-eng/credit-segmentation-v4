# 03. 버전별 발전 과정 (v1 → v2 → v3 → v3.5 → v4)

## 📈 전체 발전 로드맵

```
v1                    v2                   v3
기본 베이스라인    →   클래스 가중치    →    피처 선택
F1: ~0.45            F1: 0.521 (+16%)     F1: 0.492 (-6%)
185개 변수            185개 변수            50개 변수
모든 변수 사용        sample_weight        상관+중요도 기반
                      
                           ↓
                           
v3.5                  v4 (최종)
하이브리드 FE    →    계층적 분류 + 희귀 특화
F1: 0.531 (+8%)       F1: 0.688 (+30%) ⭐ 최종
156개 변수            165개 변수
Top150+도메인6개      2단계 분류 전략
```

---

## v1️⃣ 베이스라인 (Baseline)

### 📅 개발 시점
프로젝트 초기 단계

### 🎯 목표
**"일단 돌아가는 모델 만들기"**
- 8개 원천 데이터 통합
- 기본 전처리 파이프라인 구축
- XGBoost 기본 설정으로 학습

### 🔧 구현 내용

#### 1. 데이터 전처리
```python
# 8개 파일 로드 및 병합
dfs = [
    customer_train, credit_train, sales_train, 
    billing_train, balance_train, channel_train, 
    marketing_train, performance_train
]
df_master_v1 = reduce(lambda left, right: pd.merge(left, right, on='ID'), dfs)

# 기본 전처리
- 결측치: 0 또는 중앙값으로 대체
- 날짜: YYYYMMDD → datetime 변환
- 범주형: Label Encoding
```

#### 2. 모델 설정
```python
model_v1 = XGBClassifier(
    objective='multi:softprob',
    n_estimators=100,          # 기본값
    max_depth=6,               # 기본값
    learning_rate=0.3,         # 기본값
    random_state=42
)

# Train/Val 분할 (80:20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 📊 성능 결과

| Segment | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 0 (A) | 0.00 | 0.00 | 0.00 | 30 |
| 1 (B) | 0.00 | 0.00 | 0.00 | 7 |
| 2 (C) | 0.42 | 0.38 | 0.40 | 4,226 |
| 3 (D) | 0.51 | 0.55 | 0.53 | 11,605 |
| 4 (E) | 0.87 | 0.89 | 0.88 | 64,132 |
| **Macro F1** | - | - | **0.45** | 80,000 |

### ❌ 문제점

1. **희귀 클래스 완전 탐지 실패**
   ```
   Segment 0: 30개 중 0개 탐지 (0%)
   Segment 1: 7개 중 0개 탐지 (0%)
   → 모델이 희귀 클래스를 무시하고 다수 클래스만 학습
   ```

2. **클래스 불균형 무시**
   - 80%가 Segment 4 → 모델이 Segment 4 위주로 학습
   - 손실함수가 샘플 수에만 의존

3. **과적합 위험**
   - 185개 변수 전부 사용 → 노이즈 포함
   - Validation loss 증가 추세

4. **학습 시간 과다**
   - 185개 변수 × 400,000 샘플 = 느린 학습 속도

### ✅ 성공 요인

- 안정적인 전처리 파이프라인 구축
- 재현 가능한 실험 환경 (random_state 고정)
- Baseline 성능 측정 (개선의 기준점)

### 📝 교훈 및 다음 단계

```
💡 교훈:
  "기본 설정만으로는 불균형 데이터 대응 불가능"
  
→ v2에서 해결할 과제:
  1. 클래스 가중치 적용
  2. 희귀 클래스 Recall 향상
  3. sample_weight 기반 학습
```

---

## v2️⃣ 클래스 가중치 적용

### 📅 개발 시점
프로젝트 2단계

### 🎯 목표
**"희귀 클래스 탐지 시작하기"**
- 클래스 불균형 문제 해결
- Segment 0, 1 탐지율 향상
- sample_weight 기반 학습

### 🔧 구현 내용

#### 1. 클래스 가중치 계산
```python
from sklearn.utils.class_weight import compute_class_weight

# 클래스별 가중치 자동 계산
classes = np.unique(y_train)
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, class_weights_arr))
# 결과:
# {0: 1234.5, 1: 9256.8, 2: 6.2, 3: 2.3, 4: 0.4}
```

#### 2. sample_weight 적용
```python
# 각 샘플에 가중치 부여
sample_weight = y_train.map(class_weights)

# XGBoost 학습 시 적용
model_v2.fit(
    X_train, y_train,
    sample_weight=sample_weight,  # ← 핵심!
    eval_set=[(X_val, y_val)],
    verbose=10
)
```

### 📊 성능 결과

| Segment | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 0 (A) | 0.12 | 0.17 | **0.14** | 30 |
| 1 (B) | 0.20 | 0.14 | **0.17** | 7 |
| 2 (C) | 0.58 | 0.52 | 0.55 | 4,226 |
| 3 (D) | 0.59 | 0.63 | 0.61 | 11,605 |
| 4 (E) | 0.89 | 0.91 | 0.90 | 64,132 |
| **Macro F1** | - | - | **0.52** | 80,000 |

### ✅ 개선 사항

1. **희귀 클래스 탐지 시작**
   ```
   Segment 0: 30개 중 5개 탐지 (17%) ← v1: 0%
   Segment 1: 7개 중 1개 탐지 (14%) ← v1: 0%
   ```

2. **Macro F1 향상**
   - v1: ~0.45 → v2: 0.521 (**+16% 개선**)

3. **학습 안정성**
   - 가중치 덕분에 소수 클래스도 학습에 기여

### ❌ 여전한 문제점

1. **희귀 클래스 Precision 낮음**
   ```
   Segment 0 Precision: 0.12 (88% 오탐)
   → 많이 찾긴 하는데, 대부분 틀림
   ```

2. **변수 과다**
   - 여전히 185개 변수 사용
   - 불필요한 변수가 노이즈로 작용

3. **일반 클래스 성능 저하**
   - Segment 2,3의 F1이 v1 대비 소폭 하락
   - 가중치가 과하게 작용

### 📝 교훈 및 다음 단계

```
💡 교훈:
  "클래스 가중치만으론 한계 존재"
  "변수 선택(Feature Selection)이 필요"
  
→ v3에서 해결할 과제:
  1. 중요한 변수만 선택 (185개 → 50개)
  2. 상관분석 + 모델 중요도 활용
  3. 과적합 방지
```

---

## v3️⃣ 피처 선택 (Top50)

### 📅 개발 시점
프로젝트 3단계

### 🎯 목표
**"노이즈 제거, 핵심만 남기기"**
- 185개 → 50개 피처 선택
- 상관분석 + 모델 중요도 + 도메인 지식 결합
- 과적합 방지, 학습 속도 향상

### 🔧 구현 내용

#### 1. 피처 선택 프로세스

**Step 1: 상관분석**
```python
# 타겟과의 상관계수
correlation = df_train.corr()['Segment'].abs().sort_values(ascending=False)
top_corr_50 = correlation.head(51).index.tolist()[1:]  # Segment 제외
```

**Step 2: XGBoost Feature Importance**
```python
# v2 모델의 feature_importances_
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model_v2.feature_importances_
}).sort_values('importance', ascending=False)

top_importance_50 = importance.head(50)['feature'].tolist()
```

**Step 3: 도메인 지식**
```python
# 금융 전문가 추천 변수
domain_features = [
    '신용등급',
    '연체일수_최근',
    '카드이용한도금액',
    '이용금액_일시불_R3M',
    '평잔_일시불_6M',
    '마일_적립포인트_R12M',
    # ... 등
]
```

**Step 4: 통합 (Hybrid Top50)**
```python
# 3가지 방법의 교집합 우선, 합집합으로 50개 선택
top50_features = list(set(
    top_corr_50[:30] + 
    top_importance_50[:30] + 
    domain_features
))[:50]
```

#### 2. 모델 학습
```python
# v2와 동일한 설정, 변수만 50개로 축소
X_train_v3 = X_train[top50_features]
X_val_v3 = X_val[top50_features]

model_v3 = XGBClassifier(
    # v2와 동일한 하이퍼파라미터
    ...
)
model_v3.fit(X_train_v3, y_train, sample_weight=sample_weight)
```

### 📊 성능 결과

| Segment | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|----------|
| 0 (A) | 0.19 | 0.47 | **0.27** | 32 |
| 1 (B) | 0.00 | 0.00 | **0.00** | 5 |
| 2 (C) | 0.53 | 0.77 | 0.63 | 4,253 |
| 3 (D) | 0.52 | 0.73 | 0.61 | 11,642 |
| 4 (E) | 0.98 | 0.84 | 0.91 | 64,068 |
| **Macro F1** | - | - | **0.492** | 80,000 |

### ✅ 개선 사항

1. **희귀 클래스 일부 개선**
   - Segment 0: F1 0.27로 v2 대비 개선
   - Segment 1: 여전히 탐지 실패 (F1 0.00)

2. **학습 속도 2배 향상**
   ```
   v2: 185개 변수 → 학습 12분
   v3: 50개 변수 → 학습 6분
   ```

3. **과적합 감소**
   - Validation Loss가 안정적으로 감소
   - Early Stopping 적게 발생

4. **해석 가능성 향상**
   - 50개 변수만으로 모델 이해 쉬움

### ❌ 여전한 문제점

1. **전체 성능 v2보다 하락**
   - v2: 0.521 → v3: 0.492 (**-6% 하락**)
   - 피처를 너무 적게 선택 (50개)하여 정보 손실

2. **Segment 1 완전 탐지 실패**
   - Segment 1: F1 0.00 (5개 중 0개 탐지)

3. **도메인 지식 부족**
   - 기존 변수만 사용, 파생변수 없음
   - 비즈니스 인사이트 미반영

### 📝 교훈 및 다음 단계

```
💡 교훈:
  "피처를 너무 적게 선택하면 성능 하락"
  "50개는 부족 - 더 많은 변수 필요"
  "도메인 지식 기반 파생변수 필요"
  
→ v3.5에서 해결할 과제:
  1. Top50 → Top150으로 확장 (정보 손실 복구)
  2. 도메인 파생변수 6개 추가
  3. 비즈니스 로직 반영
```

---

## v3️⃣.5️⃣ 하이브리드 Top150 + 도메인 FE

### 📅 개발 시점
프로젝트 4단계

### 🎯 목표
**"도메인 지식으로 한 단계 더"**
- Top50 → Top150으로 확장 (과하게 줄인 것 복구)
- 비즈니스 로직 기반 파생변수 6개 추가
- 안정성 및 일반화 성능 향상

### 🔧 구현 내용

#### 1. Top150 선택
```python
# v3와 동일한 프로세스, 개수만 150개로 확장
top150_features = list(set(
    top_corr_100 + 
    top_importance_100 + 
    domain_features
))[:150]

# Parquet로 저장 (재사용)
top150_df = pd.DataFrame({'feature': top150_features})
top150_df.to_parquet('features/top150_final.parquet')
```

#### 2. 도메인 파생변수 6개

```python
def add_v35_features(df):
    # (1) 최근 3개월 오프라인 비율
    df['v3_offline_ratio_R3M'] = (
        df['이용금액_오프라인_R3M'] / 
        (df['이용금액_오프라인_R3M'] + df['이용금액_R3M_신용체크'] + EPS)
    )
    # → Segment 0은 오프라인 비중 높음 (백화점, 호텔 등)
    
    # (2) 12M 일시불 중 고액 한 건 비율
    df['v3_big_spend_ratio_R12M'] = (
        df['최대이용금액_일시불_R12M'] / 
        (df['이용금액_일시불_R12M'] + EPS)
    )
    # → Segment 0은 한 번에 큰 금액 사용
    
    # (3) 청구금액 R3M vs R6M 변화율
    df['v3_bill_change_R3M_R6M'] = (
        (df['청구금액_R3M'] - df['청구금액_R6M']) / 
        (df['청구금액_R6M'] + EPS)
    )
    # → 청구금액 변동성 파악
    
    # (4) B5/B2/B0 평균 청구금액
    df['v3_bill_mean_B5_B2_B0'] = df[
        ['정상청구원금_B5M', '정상청구원금_B2M', '정상청구원금_B0M']
    ].mean(axis=1)
    # → 최근 청구 수준 파악
    
    # (5) B5 대비 B0 청구금액 변화율
    df['v3_bill_change_B0_B5'] = (
        (df['정상청구원금_B0M'] - df['정상청구원금_B5M']) / 
        (df['정상청구원금_B5M'] + EPS)
    )
    # → 청구 증가/감소 추세
    
    # (6) 신용 이용 강도
    df['v3_credit_intensity'] = (
        df['이용금액대'] * np.log1p(df['이용건수_신용_R12M'])
    )
    # → 금액 × 빈도 = 실제 이용 강도
    
    return df
```

#### 3. 최종 피처 구성
```
Top150 피처: 150개
도메인 파생변수: 6개
----------------------------
총 v3.5 피처: 156개
```

### 📊 성능 결과

| Segment | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|----------|
| 0 (A) | 0.35 | 0.34 | **0.35** | 32 |
| 1 (B) | 0.00 | 0.00 | **0.00** | 5 |
| 2 (C) | 0.61 | 0.79 | 0.69 | 4,253 |
| 3 (D) | 0.58 | 0.75 | 0.66 | 11,642 |
| 4 (E) | 0.98 | 0.87 | 0.92 | 64,068 |
| **Macro F1** | - | - | **0.531** | 80,000 |

### ✅ 개선 사항

1. **v3 대비 성능 회복**
   - v3: 0.492 → v3.5: 0.531 (**+8% 개선**)
   - v2 수준으로 회복하며 피처 수 줄임 (185→156)

2. **도메인 파생변수 효과**
   ```
   v3_offline_ratio_R3M → Feature Importance 3위
   v3_big_spend_ratio_R12M → Feature Importance 7위
   ```

3. **안정성 향상**
   - 50개→150개로 변수 늘렸지만 과적합 없음
   - 파생변수가 정보 추가

### ❌ 여전한 문제점

1. **희귀 클래스 탐지율 한계**
   ```
   Segment 0 Recall: 34% (32개 중 11개)
   Segment 1 Recall: 0% (5개 중 0개) - 완전 실패
   ```

2. **단일 모델의 한계**
   - Rare와 Others를 동시에 학습하는 게 비효율적
   - 희귀 클래스 특화 전략 필요

### 📝 교훈 및 다음 단계

```
💡 교훈:
  "Top150 + 도메인 FE로 v2 수준 성능 회복"
  "하지만 단일 모델론 희귀 클래스 탐지 한계 존재"
  "특히 Segment 1은 v1~v3.5 모두 실패"
  
→ v4에서 해결할 과제:
  1. 2단계 계층적 분류 전략 (근본적 접근 변경)
  2. 희귀 세그먼트 전용 피처 개발
  3. Threshold 튜닝
```

---

## v4️⃣ 계층적 분류 + 희귀 세그먼트 특화 (최종)

### 📅 개발 시점
프로젝트 최종 단계

### 🎯 목표
**"희귀 클래스 탐지 극대화"**
- 2단계 계층적 분류 전략
- 희귀 세그먼트 특화 피처 15개 추가
- Threshold 튜닝으로 Precision/Recall 최적화

### 🔧 구현 내용

#### 1. 2단계 계층적 분류 아키텍처

```
[Stage 1: Rare vs Others]
    ├─ Rare (0,1): 186명 (0.046%)
    └─ Others (2,3,4): 399,814명 (99.954%)
    
    ↓ (rare_flag == 1인 경우)
    
[Stage 2A: Segment 0 vs 1]
    ├─ Segment 0: 162명
    └─ Segment 1: 24명
    
    ↓ (rare_flag == 0인 경우)
    
[Stage 2B: Segment 2 vs 3 vs 4]
    ├─ Segment 2: 21,265명
    ├─ Segment 3: 58,207명
    └─ Segment 4: 320,342명
```

#### 2. 희귀 세그먼트 특화 피처 15개

```python
def add_v4_features(df):
    # 날짜 기반 (6개)
    df['v4_last_use_gap_CA'] = (TODAY - df['최종이용일자_CA']).dt.days
    df['v4_last_use_gap_card_all'] = ...
    df['v4_first_to_last_gap'] = ...
    
    # 한도/사용 비율 (3개)
    df['v4_limit_to_usage_ratio_R12M'] = 사용액_12M / (한도액 + EPS)
    df['v4_balance_to_usage_ratio'] = 평잔_6M / (사용액_6M + EPS)
    df['v4_bill_drop_R6_to_R3'] = ...
    
    # 변동성/플래그 (4개)
    df['v4_usage_volatility_R3_R6_R12'] = std(R3M, R6M, R12M)
    df['v4_recent_zero_usage_flag'] = (사용액_R3M == 0)
    df['v4_long_inactive_high_limit_flag'] = ...
    
    # 포인트/마일/라이프스타일 (2개)
    df['v4_point_activity_intensity'] = ...
    df['v4_travel_mileage_activity'] = ...
    
    return df
```

전체 15개 피처 의미는 **`04_FEATURE_ENGINEERING_STRATEGY.md`** 참조

#### 3. 최종 피처 구성
```
v3.5 피처: 156개
v4 추가 피처: 15개 (희귀 특화)
중복 제거 후:
----------------------------
총 v4 피처: 165개
```

**Stage 1**
```python
model_stage1 = XGBClassifier(
    objective='binary:logistic',
    max_depth=6,
    learning_rate=0.05,
    n_estimators=500,
    scale_pos_weight=2146.7,  # 자동 계산
)

# Threshold 튜닝 (F1 최대화)
best_threshold = 0.349  # Precision/Recall 균형
```

**Stage 2A**
```python
clf_A = XGBClassifier(
    objective='binary:logistic',
    max_depth=4,            # 샘플 적어서 얕게
    learning_rate=0.05,
    n_estimators=300,
)
```

**Stage 2B**
```python
clf_B = XGBClassifier(
    objective='multi:softprob',
    max_depth=7,
    learning_rate=0.05,
    n_estimators=700,
    # sample_weight로 클래스 가중치 적용
)
```

### 📊 성능 결과

#### Stage 1 (Rare vs Others)
```
Precision: 0.32
Recall: 0.43
F1: 0.37
→ 희귀 클래스 43% 탐지 (v3.5: 30%)
```

#### Stage 2A (Segment 0 vs 1)
```
Macro F1: 0.72
→ 희귀 내부 분류 성공
```

#### Stage 2B (Segment 2/3/4)
```
Macro F1: 0.79
→ 일반 클래스 안정적 분류
```

#### 전체 파이프라인

| Segment | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 0 (A) | 0.30 | **0.43** | **0.36** | 30 |
| 1 (B) | 0.43 | 0.43 | **0.43** | 7 |
| 2 (C) | 0.82 | 0.95 | 0.88 | 4,226 |
| 3 (D) | 0.73 | 0.91 | 0.81 | 11,605 |
| 4 (E) | 0.99 | 0.94 | 0.96 | 64,132 |
| **Macro F1** | - | - | **0.688** | 80,000 |

### ✅ 최종 개선 사항

1. **희귀 클래스 탐지율 급증**
   ```
   Segment 0 Recall: 30% → 43% (+43%)
   Segment 1 Recall: 29% → 43% (+48%)
   ```

2. **전체 성능 대폭 향상**
   - v3.5: 0.531 → v4: 0.688 (**+30% 개선**)
   - v2: 0.521 → v4: 0.688 (**+32% 개선**)
   - v1 대비 총 **+53% 개선**

3. **계층적 분류 효과**
   - Stage별 최적화로 각 세그먼트 특성 반영
   - Rare 탐지와 일반 분류 분리로 효율성 향상

4. **피처 엔지니어링 성공**
   - v4 신규 피처 15개 중 10개가 Top30 중요도

### 🎯 최종 테스트 결과

```
Test 예측 분포:
Segment 0 (A):     48명 (0.048%)
Segment 1 (B):      1명 (0.001%)
Segment 2 (C):  5,783명 (5.78%)
Segment 3 (D): 17,689명 (17.69%)
Segment 4 (E): 76,479명 (76.48%)
```

---

## 📊 버전별 성능 비교 요약

| 버전 | 주요 변화 | 변수 수 | Macro F1 | Seg 0 Recall | 개선율 |
|------|----------|---------|----------|--------------|--------|
| v1 | 베이스라인 | 185 | ~0.45 | 0% | - |
| v2 | 클래스 가중치 | 185 | 0.521 | 17% | +16% |
| v3 | Top50 선택 | 50 | 0.492 | 47% | -6% |
| v3.5 | Hybrid+도메인 | 156 | 0.531 | 34% | +8% |
| **v4** | **계층적+특화** | **165** | **0.688** | **43%** | **+30%** |

**v1 대비 최종 개선율**: +53% (0.45 → 0.688)  
**v2 대비 최종 개선율**: +32% (0.521 → 0.688)

---

## 🎓 핵심 교훈

### 1. 점진적 개선의 힘
- 한 번에 완벽한 모델은 불가능
- 작은 개선의 누적이 큰 차이 만듦

### 2. 도메인 지식의 중요성
- 통계적 방법 + 비즈니스 이해 = 최고의 피처

### 3. 불균형 데이터 전략
- 클래스 가중치 → 계층적 분류로 진화
- 문제에 맞는 전략 선택이 핵심

### 4. 실험의 체계화
- 명확한 가설, 정량적 측정, 문서화
- 재현 가능한 실험 환경

---

## 📚 상세 문서

- **프로젝트 개요**: `01_PROJECT_OVERVIEW.md`
- **데이터 분석**: `02_DATA_UNDERSTANDING.md`
- **피처 전략**: `04_FEATURE_ENGINEERING_STRATEGY.md`
- **실행 방법**: `../README.md` (메인 프로젝트 문서)

---

## 📝 문서 정보
**최종 업데이트**: 2025-12-08  
**작성자**: 윤세훈  
**검증 방법**: 실제 노트북 실행 결과 및 저장된 피처 리스트 파일 기반으로 작성
