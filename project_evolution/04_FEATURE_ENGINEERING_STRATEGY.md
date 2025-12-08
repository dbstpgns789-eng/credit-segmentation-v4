# 04. 피처 엔지니어링 전략 및 철학

## 🎯 피처 엔지니어링의 목표

```
"적은 수의 의미 있는 피처로 최대 성능 달성"

- 노이즈 제거: 불필요한 변수는 과적합 유발
- 정보 압축: 여러 변수의 핵심을 하나로
- 도메인 반영: 비즈니스 로직을 수학적으로 표현
- 해석 가능성: 모델의 판단 근거를 이해 가능하게
```

---

## 📐 피처 엔지니어링 3단계 전략

### 🔵 1단계: 피처 선택 (Feature Selection)

**"185개 중 무엇을 남길 것인가?"**

#### 방법 1: 상관분석 (Correlation Analysis)
```python
# 타겟과의 피어슨 상관계수
correlation = df.corr()['Segment'].abs()
top_corr_features = correlation.sort_values(ascending=False).head(100)

# 예시 결과
"""
마일_적립포인트_R12M           0.42  ← 강한 양의 상관
이용금액_일시불_R12M           0.38
신용등급                      -0.35  ← 강한 음의 상관
이용건수_신용_R12M             0.32
"""
```

**장점**: 빠르고 명확  
**단점**: 비선형 관계 포착 못함, 변수 간 상호작용 무시

#### 방법 2: 모델 기반 중요도 (Model-based Importance)
```python
# XGBoost Feature Importance
model.fit(X_train, y_train)
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# 예시 결과 (Top 10)
"""
1. 평잔_일시불_6M                    0.087
2. 이용금액_일시불_R3M               0.065
3. v3_offline_ratio_R3M             0.058  ← 파생변수!
4. 신용등급                          0.052
5. 카드이용한도금액                  0.048
6. 이용건수_체크_R12M                0.045
7. v3_big_spend_ratio_R12M          0.042  ← 파생변수!
8. 잔액_신판ca최대한도소진율_r6m    0.039
9. 한도증액후경과월                  0.037
10. 소지카드수_유효_신용             0.035
"""
```

**장점**: 실제 모델 학습에 기여한 변수 파악  
**단점**: 모델 의존적, 학습 데이터에 과적합 가능

#### 방법 3: 도메인 지식 (Domain Knowledge)
```python
# 금융 전문가 추천 변수 (반드시 포함)
domain_must_have = [
    '신용등급',              # 고객 신용도 핵심
    '연체일수_최근',         # 리스크 지표
    '카드이용한도금액',      # 고객 등급 반영
    '마일_적립포인트_R12M',  # 고급 고객 지표
    '이용금액_일시불_R3M',   # 최근 사용 패턴
    '평잔_일시불_6M',        # 잔액 수준
]
```

**장점**: 비즈니스 인사이트 반영, 해석 가능성  
**단점**: 주관적, 데이터로 검증 필요

#### 최종: Hybrid Top150 선택
```python
# 3가지 방법 통합
candidates = set(
    top_corr_features.index.tolist()[:100] +     # 상관분석 상위 100
    importance.head(100)['feature'].tolist() +   # 중요도 상위 100
    domain_must_have                              # 도메인 필수
)

# 최종 150개 선택 (중복 제거 + 추가 필터링)
top150 = list(candidates)[:150]

# 저장 (재사용)
pd.DataFrame({'feature': top150}).to_parquet('top150_final.parquet')
```

---

### 🟢 2단계: 피처 생성 (Feature Engineering)

**"없는 정보를 만들어낸다"**

#### 패턴 1: 비율 (Ratio)
```python
# 왜? 절대값보다 상대적 비율이 의미 있음

# (1) 오프라인 사용 비율
df['v3_offline_ratio_R3M'] = (
    df['이용금액_오프라인_R3M'] / 
    (df['이용금액_오프라인_R3M'] + df['이용금액_온라인_R3M'] + EPS)
)
# Segment 0: 0.8 (오프라인 80%)
# Segment 4: 0.3 (오프라인 30%)

# (2) 한 건당 고액 사용 비율
df['v3_big_spend_ratio_R12M'] = (
    df['최대이용금액_일시불_R12M'] / 
    (df['이용금액_일시불_R12M'] + EPS)
)
# Segment 0: 0.6 (한 번에 60% 사용)
# Segment 4: 0.05 (한 번에 5% 사용)

# (3) 한도 대비 사용 비율
df['v4_limit_to_usage_ratio_R12M'] = (
    df['이용금액_일시불_R12M'] / 
    (df['카드이용한도금액'] + EPS)
)
# Segment 0: 0.1 (한도의 10% 사용)
# Segment 4: 0.7 (한도의 70% 사용)
```

**철학**: "절대값은 개인차가 크지만, 비율은 패턴을 드러낸다"

#### 패턴 2: 변화율 (Change Rate)
```python
# 왜? 트렌드가 세그먼트를 결정

# (1) 청구금액 변화율
df['v3_bill_change_R3M_R6M'] = (
    (df['청구금액_R3M'] - df['청구금액_R6M']) / 
    (df['청구금액_R6M'] + EPS)
)
# Segment 0: +50% (증가 추세, 계절적 고액 사용)
# Segment 4: -5% (거의 일정)

# (2) 월별 청구금액 변화
df['v3_bill_change_B0_B5'] = (
    (df['정상청구원금_B0M'] - df['정상청구원금_B5M']) / 
    (df['정상청구원금_B5M'] + EPS)
)

# (3) 청구금액 하락율
df['v4_bill_drop_R6_to_R3'] = (
    (df['청구금액_R6M'] - df['청구금액_R3M']) / 
    (df['청구금액_R6M'] + EPS)
)
```

**철학**: "현재 상태보다 변화 방향이 중요하다"

#### 패턴 3: 변동성 (Volatility)
```python
# 왜? 사용 패턴의 안정성/불규칙성 파악

# (1) 사용액 변동성
df['v4_usage_volatility_R3_R6_R12'] = np.std([
    df['이용금액_일시불_R3M'],
    df['이용금액_일시불_R6M'],
    df['이용금액_일시불_R12M']
], axis=0)
# Segment 0: 높음 (불규칙 사용)
# Segment 4: 낮음 (규칙적 사용)
```

**철학**: "불규칙성도 하나의 특징이다"

#### 패턴 4: 조합 (Combination)
```python
# 왜? 단일 변수로 설명 안 되는 복합 개념

# (1) 신용 이용 강도
df['v3_credit_intensity'] = (
    df['이용금액대'] * np.log1p(df['이용건수_신용_R12M'])
)
# 금액 × log(1+빈도) = 실제 활동성

# (2) 포인트 활동 강도
df['v4_point_activity_intensity'] = (
    (df['포인트_적립포인트_R12M'] + df['포인트_이용포인트_R12M']) / 
    (df['이용금액_일시불_R12M'] + df['이용금액_할부_R12M'] + EPS)
)
# 포인트 총량 / 사용액 = 포인트 활용도

# (3) 마일리지 활동
df['v4_travel_mileage_activity'] = (
    (df['마일_적립포인트_R12M'] + df['마일_이용포인트_R12M']) / 
    (df['이용금액_일시불_R12M'] + EPS)
)
```

**철학**: "1+1이 3이 될 수 있다"

#### 패턴 5: 플래그 (Boolean Flag)
```python
# 왜? 극단적 상황을 명확히 표시

# (1) 최근 사용 없음
df['v4_recent_zero_usage_flag'] = (
    (df['이용금액_일시불_R3M'] + 
     df['이용금액_할부_R3M'] + 
     df['이용금액_CA_R3M']) == 0
).astype('int8')
# Segment 0: 많음 (분기 1~2회만 사용)
# Segment 4: 거의 없음

# (2) 장기 미사용 + 고한도
df['v4_long_inactive_high_limit_flag'] = (
    (df['카드이용한도금액'] > 3_000_000) &      # 한도 300만원+
    (df['이용금액_R12M'] < 100_000) &           # 12개월 10만원 미만
    (df['최종이용일자_gap'] > 365)               # 1년 이상 미사용
).astype('int8')
# Segment 0 특징: "잠자는 고한도 카드"

# (3) 연체 플래그
df['v4_arrears_recent_flag'] = (
    df['연체일수_최근'] > 30
).astype('int8')

# (4) 카드론 상환 완료
df['v4_cardloan_cleanup_flag'] = (
    (df['카드론이용금액_누적'] > 1_000_000) &    # 과거 카드론 사용
    (df['잔액_카드론_B0M'] == 0) &               # 현재 잔액 0
    (df['최종이용일자_카드론_gap'] > 365)        # 1년 이상 사용 안 함
).astype('int8')
```

**철학**: "극단은 단순한 0/1로 표현하는 게 강력하다"

#### 패턴 6: 시간 경과 (Time Elapsed)
```python
# 왜? "얼마나 지났는가"도 중요한 정보

TODAY = pd.Timestamp("2024-12-31")

# (1) CA 최종 이용 후 경과일
df['v4_last_use_gap_CA'] = (
    TODAY - pd.to_datetime(df['최종이용일자_CA'], format='%Y%m%d')
).dt.days
# Segment 0: 1000+ 일 (거의 사용 안 함)
# Segment 4: 30-90일 (간헐적 사용)

# (2) 전체 카드 최종 이용 후 경과일
df['v4_last_use_gap_card_all'] = ...

# (3) 가입~최종 이용 기간
df['v4_first_to_last_gap'] = (
    df['최종이용일자_기본'] - df['입회일자']
).dt.days
# Segment 0: 짧음 (최근 가입, 단기 고액 사용)
# Segment 4: 길음 (오래 사용, 꾸준함)
```

**철학**: "시간의 흐름은 행동 패턴의 역사다"

---

### 🔴 3단계: 피처 검증 (Feature Validation)

**"만든 피처가 정말 유용한가?"**

#### 검증 1: Feature Importance 분석
```python
# v4 피처 중요도 확인
v4_features = [f for f in importance['feature'] if f.startswith('v4_')]
v4_importance = importance[importance['feature'].isin(v4_features)]

print(v4_importance.sort_values('importance', ascending=False))
"""
v4_limit_to_usage_ratio_R12M         0.058  (3위!)
v4_long_inactive_high_limit_flag     0.045  (7위!)
v4_balance_to_usage_ratio            0.039  (12위)
v4_point_activity_intensity          0.035  (15위)
...

→ 15개 중 10개가 Top 30 진입 ✅
"""
```

#### 검증 2: 세그먼트별 분포 비교
```python
# 피처가 세그먼트를 잘 구분하는지 확인
for feature in v4_features:
    print(f"\n{feature}")
    print(df.groupby('Segment')[feature].describe())

"""
v4_limit_to_usage_ratio_R12M
         mean    std    min    25%    50%    75%    max
Segment
0        0.08   0.05   0.01   0.04   0.07   0.11   0.25  ← 낮음
1        0.12   0.08   0.02   0.06   0.10   0.15   0.35
2        0.45   0.20   0.10   0.30   0.45   0.60   0.85
3        0.62   0.18   0.25   0.50   0.62   0.75   0.95
4        0.78   0.15   0.40   0.68   0.80   0.90   1.00  ← 높음

→ 명확히 구분됨! ✅
"""
```

#### 검증 3: 단일 피처 성능 테스트
```python
# 피처 하나만으로 얼마나 예측 가능한지
from sklearn.tree import DecisionTreeClassifier

for feature in v4_features:
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train[[feature]], y_train)
    score = clf.score(X_val[[feature]], y_val)
    print(f"{feature}: {score:.3f}")

"""
v4_limit_to_usage_ratio_R12M         0.72  ← 단독으로도 강력!
v4_long_inactive_high_limit_flag     0.68
v4_balance_to_usage_ratio            0.65
...

→ 개별 피처의 판별력 확인 ✅
"""
```

#### 검증 4: 다중공선성 체크
```python
# 새 피처가 기존 피처와 중복되는지 확인
from scipy.stats import spearmanr

# v4 피처 간 상관계수
v4_corr = df[v4_features].corr()
high_corr = (v4_corr.abs() > 0.8) & (v4_corr.abs() < 1.0)

if high_corr.any().any():
    print("⚠️ 높은 상관관계 발견:")
    print(v4_corr[high_corr])
else:
    print("✅ 다중공선성 없음")
```

---

## 🎯 피처 엔지니어링 체크리스트

### ✅ 좋은 피처의 조건

1. **판별력 (Discriminative Power)**
   - 세그먼트별로 명확한 분포 차이
   - Feature Importance 상위권

2. **해석 가능성 (Interpretability)**
   - 비즈니스적으로 설명 가능
   - "왜 이 피처가 중요한가?" 답변 가능

3. **안정성 (Stability)**
   - Train/Val 분포 유사
   - 이상치에 민감하지 않음

4. **독립성 (Independence)**
   - 기존 피처와 중복 최소화
   - 새로운 정보 제공

5. **효율성 (Efficiency)**
   - 계산 비용 적정
   - 실시간 서빙 가능

### ❌ 나쁜 피처의 예

1. **Data Leakage**
   ```python
   # 🚫 절대 금지!
   df['future_segment'] = df['Segment'].shift(-1)  # 미래 정보 사용
   df['test_average'] = test['Segment'].mean()     # Test 정보 사용
   ```

2. **타겟과 직접 관련**
   ```python
   # 🚫 너무 직접적
   df['is_segment_0'] = (df['Segment'] == 0)  # 타겟 그대로
   ```

3. **과도한 과적합**
   ```python
   # 🚫 Train에만 맞춤
   df['customer_id_encoded'] = LabelEncoder().fit_transform(df['ID'])
   # ID는 고유값, 일반화 불가능
   ```

4. **의미 없는 조합**
   ```python
   # 🚫 논리적 근거 없음
   df['random_ratio'] = df['나이'] / (df['마일_적립'] + 1)
   # 나이와 마일의 비율? 의미 없음
   ```

---

## 📊 v4 피처 엔지니어링 성과

### 최종 피처 구성 (167개)

```
[v3.5 베이스: 156개]
├─ Top150 Hybrid: 150개
└─ v3.5 도메인 FE: 6개

[v4 신규 추가: 15개]
├─ 시간 경과: 3개
├─ 비율: 3개
├─ 변동성: 1개
├─ 플래그: 4개
├─ 조합: 2개
└─ 온/오프라인: 2개
------------------------
총 167개 피처 (중복 제거 후)
```

### 피처별 중요도 Top 20

| 순위 | 피처명 | 타입 | Importance |
|------|--------|------|------------|
| 1 | 이용개월수_결제일_R6M | 원천 | 0.087 |
| 2 | 평잔_일시불_6M | 원천 | 0.078 |
| 3 | **v4_limit_to_usage_ratio_R12M** | **v4 신규** | **0.065** |
| 4 | 이용건수_체크_R12M | 원천 | 0.058 |
| 5 | 이용금액_일시불_R3M | 원천 | 0.055 |
| 6 | **v4_balance_to_usage_ratio** | **v4 신규** | **0.052** |
| 7 | 소지카드수_유효_신용 | 원천 | 0.048 |
| 8 | 잔액_신판ca최대한도소진율_r6m | 원천 | 0.045 |
| 9 | 월중평잔_일시불 | 원천 | 0.042 |
| 10 | 한도증액후경과월 | 원천 | 0.039 |
| 11 | 이용건수_오프라인_R3M | 원천 | 0.038 |
| 12 | 청구금액_R3M | 원천 | 0.036 |
| 13 | 이용개월수_신판_R3M | 원천 | 0.035 |
| 14 | 최종이용일자_할부 | 원천 | 0.034 |
| 15 | 이용건수_일시불_B0M | 원천 | 0.033 |
| 16 | 이용카드수_신용체크 | 원천 | 0.032 |
| 17 | **v4_long_inactive_high_limit_flag** | **v4 신규** | **0.031** |
| 18 | 입회일자_신용 | 원천 | 0.030 |
| 19 | 이용건수_신판_R12M | 원천 | 0.029 |
| 20 | **v3_offline_ratio_R3M** | **v3.5** | **0.028** |

**v4 신규 피처가 Top 20 중 3개 차지! (15%, 6%, 17위)**

---

## 🔍 피처 엔지니어링 철학

### 1. "Less is More"
- 많은 변수보다 의미 있는 소수 변수
- 185개 → 50개 → 156개 → 167개 (최종)

### 2. "Domain First, Data Second"
- 데이터 분석보다 비즈니스 이해가 먼저
- 통계는 검증 도구

### 3. "Simplicity & Interpretability"
- 복잡한 피처는 디버깅 어려움
- 간단한 비율/플래그가 강력

### 4. "Iterate & Validate"
- 한 번에 완벽한 피처 불가능
- 만들고 → 테스트 → 개선 반복

---

## 📚 참고 문서

- **데이터 이해**: `02_DATA_UNDERSTANDING.md` - 원천 데이터 상세
- **버전 발전**: `03_VERSION_EVOLUTION.md` - v1~v4 피처 진화 과정
- **전체 개요**: `01_PROJECT_OVERVIEW.md` - 프로젝트 타임라인

---

## 📝 문서 정보
**최종 업데이트**: 2025-12-08  
**작성자**: 윤세훈  
**기반**: v1~v4 노트북 코드 및 실험 결과
