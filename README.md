# 신용카드 고객 세그먼트 분류 프로젝트 (v4)

## 📌 프로젝트 개요
신용카드 고객 데이터를 활용하여 5개 세그먼트(A~E, 또는 0~4)로 고객을 분류하는 머신러닝 프로젝트입니다.

**핵심 전략**: 2단계 계층적 분류 (Hierarchical Classification)
- **Stage 1**: Rare(희귀 세그먼트 0,1) vs Others(일반 세그먼트 2,3,4) 이진 분류
- **Stage 2A**: Rare 내부에서 Segment 0 vs 1 분류
- **Stage 2B**: Others 내부에서 Segment 2 vs 3 vs 4 분류

---

## 📁 디렉토리 구조

```
final_submission/
│
├── README.md                          # 본 문서
│
├── notebooks/                         # 실행 노트북
│   ├── step1_build_v4_features.ipynb  # 피처 엔지니어링 및 데이터 생성
│   └── step2_train_and_predict.ipynb  # 모델 학습 및 예측
│
├── data/                              # 학습/테스트 데이터
│   ├── df_master_v4_train.parquet     # v4 피처 포함 학습 데이터 (165개)
│   ├── df_master_v4_test.parquet      # v4 피처 포함 테스트 데이터 (165개)
│   └── features/                      # 피처 리스트
│       ├── top150_final.parquet       # Hybrid Top150 피처 목록
│       └── v4_feature_list.csv        # v4 최종 피처 165개
│
├── models/                            # 학습된 모델 (step2 실행 후 생성)
│   ├── model_stage1_rare_vs_others.pkl
│   ├── model_stage2A_seg01.pkl
│   ├── model_stage2B_seg234.pkl
│   └── label_encoder_234.pkl
│
└── results/                           # 예측 결과 (step2 실행 후 생성)
    ├── v4_test_predictions.csv        # 최종 제출 파일 (CSV)
    └── v4_test_predictions.parquet    # 최종 제출 파일 (Parquet)
```

---

## 🚀 실행 방법

### 1️⃣ 환경 설정
```bash
# 필수 라이브러리 설치
pip install pandas numpy xgboost scikit-learn pyarrow
```

### 2️⃣ Step 1: 피처 엔지니어링 및 데이터 생성
**파일**: `notebooks/step1_build_v4_features.ipynb`

**역할**:
- v1 전처리 데이터(`df_master_preprocessed_v1_train/test.parquet`)를 로드
- v3.5 피처 생성 (Hybrid Top150 + 도메인 파생변수 6개)
- v4 신규 피처 15개 추가
- 최종 데이터 저장: `data/df_master_v4_train.parquet`, `data/df_master_v4_test.parquet`

**실행**:
- Jupyter Notebook에서 모든 셀 실행

**생성 파일**:
- `data/df_master_v4_train.parquet` (165개 피처 + Segment)
- `data/df_master_v4_test.parquet` (165개 피처)

---

### 3️⃣ Step 2: 모델 학습 및 예측
**파일**: `notebooks/step2_train_and_predict.ipynb`

**역할**:
- Step1에서 생성한 v4 데이터 로드
- 2단계 계층적 분류 모델 학습
  - **Stage 1**: XGBoost 이진 분류 (rare vs others) + Threshold 튜닝
  - **Stage 2A**: XGBoost 이진 분류 (Segment 0 vs 1)
  - **Stage 2B**: XGBoost 다중 분류 (Segment 2 vs 3 vs 4) + 클래스 가중치
- Test 데이터 예측 및 제출 파일 생성

**실행**:
- Jupyter Notebook에서 모든 셀 실행

**생성 파일**:
- `models/model_stage1_rare_vs_others.pkl`
- `models/model_stage2A_seg01.pkl`
- `models/model_stage2B_seg234.pkl`
- `models/label_encoder_234.pkl`
- `results/v4_test_predictions.csv` ⭐ **최종 제출 파일**
- `results/v4_test_predictions.parquet`

---

## 📊 v4 피처 엔지니어링 상세

### v3.5 피처 (156개)
1. **Hybrid Top150**: 150개 중요 피처 (상관분석 + 도메인 지식 + 모델 중요도 기반)
2. **도메인 파생변수 6개**:
   - `v3_offline_ratio_R3M`: 최근 3개월 오프라인 비율
   - `v3_big_spend_ratio_R12M`: 12개월 일시불 중 고액 한 건 비율
   - `v3_bill_change_R3M_R6M`: 청구금액 R3M vs R6M 변화율
   - `v3_bill_mean_B5_B2_B0`: B5/B2/B0 평균 청구금액
   - `v3_bill_change_B0_B5`: B5 대비 B0 청구금액 변화율
   - `v3_credit_intensity`: 신용 이용 강도 (이용금액대 × log(1+신용건수))

### v4 신규 피처 (15개)
**희귀 세그먼트(0,1) 특성 포착을 위한 고급 파생변수**:

1. `v4_last_use_gap_CA`: CA(현금서비스) 최종 이용 이후 경과일
2. `v4_last_use_gap_card_all`: 전체 카드 최종 이용 이후 경과일
3. `v4_first_to_last_gap`: 가입일 ~ 최종 이용일 기간
4. `v4_limit_to_usage_ratio_R12M`: 12개월 사용액 / 한도액 비율
5. `v4_balance_to_usage_ratio`: 평균잔액 / 6개월 사용액 비율
6. `v4_bill_drop_R6_to_R3`: 청구금액 감소율 (R6M → R3M)
7. `v4_usage_volatility_R3_R6_R12`: 사용액 변동성 (R3M/R6M/R12M 표준편차)
8. `v4_recent_zero_usage_flag`: 최근 3개월 사용액 0 여부
9. `v4_long_inactive_high_limit_flag`: 장기 미사용 + 고한도 플래그
10. `v4_point_activity_intensity`: 포인트 활동 강도
11. `v4_travel_mileage_activity`: 마일리지 활동 강도
12. `v4_lifestyle_auto_payment_flag`: 자동이체 부재 플래그
13. `v4_arrears_recent_flag`: 최근 연체 플래그 (30일 이상)
14. `v4_cardloan_cleanup_flag`: 카드론 상환 완료 플래그
15. `v4_online_offline_usage_ratio_R6M`: 온라인 vs 오프라인 사용 비율

**최종**: v3.5(156개) + v4(15개) → **165개 피처** (중복 제거 후)

---

## 🎯 모델 구조 및 하이퍼파라미터

### Stage 1: Rare vs Others (이진 분류)
- **알고리즘**: XGBoost Binary Classification
- **목적**: Segment 0,1 (희귀) vs 2,3,4 (일반) 구분
- **주요 설정**:
  ```python
  max_depth=6
  learning_rate=0.05
  n_estimators=500
  scale_pos_weight=자동계산 (클래스 불균형 보정)
  ```
- **Threshold 튜닝**: F1-Score 최댓값 기준으로 최적 임계값 선택

### Stage 2A: Segment 0 vs 1 (희귀 내부)
- **알고리즘**: XGBoost Binary Classification
- **주요 설정**:
  ```python
  max_depth=4
  learning_rate=0.05
  n_estimators=300
  ```

### Stage 2B: Segment 2 vs 3 vs 4 (일반 내부)
- **알고리즘**: XGBoost Multi-class Classification
- **주요 설정**:
  ```python
  max_depth=7
  learning_rate=0.05
  n_estimators=700
  sample_weight=클래스 가중치 적용
  ```

---

## 📈 성능 평가 지표

- **주 평가지표**: Macro F1-Score
- **보조지표**: Precision, Recall, Confusion Matrix (클래스별)

**Validation 결과** (step2 실행 후 확인):
- Stage1 F1-Score: [실행 후 확인]
- Stage2A Macro F1: [실행 후 확인]
- Stage2B Macro F1: [실행 후 확인]
- **Overall Macro F1**: [실행 후 확인]

---

## 📦 제출 파일

**파일**: `results/v4_test_predictions.csv`

**형식**:
```csv
ID,Segment_pred,Segment_pred_label
TRAIN_000002,3,D
TRAIN_000007,2,C
...
```

**컬럼 설명**:
- `ID`: 고객 ID
- `Segment_pred`: 예측된 세그먼트 (0~4)
- `Segment_pred_label`: 예측된 세그먼트 레이블 (A~E)

---

## 🔧 트러블슈팅

### 문제 1: 메모리 부족
**해결책**: step1에서 dtype 최적화 (float32, int8) 적용됨

### 문제 2: 파일 경로 오류
**해결책**: 
- `notebooks/` 폴더 내 노트북에서 상대 경로 사용
- 프로젝트 루트: `C:\Users\User\전산통계프로젝트`

### 문제 3: 이전 프로젝트 파일 참조 오류
**해결책**: 
- `data/features/` 폴더에 필요한 파일 복사 완료
- `top150_final.parquet`, `v4_feature_list.csv` 포함

---

## 📝 버전 히스토리

- **v1**: XGBoost 기본 베이스라인
- **v2**: 클래스 가중치 추가
- **v3**: Hybrid Top50 피처 선택
- **v3.5**: Top150 + 도메인 파생변수 6개
- **v4** ⭐ **현재 버전**: 
  - v3.5 + 희귀 세그먼트 특화 피처 15개
  - 2단계 계층적 분류 전략
  - Threshold 튜닝 및 클래스 가중치 최적화

---

---

## 📝 문서 정보
**최종 수정일**: 2025-12-08  
**작성자**: 윤세훈  
**검증**: 실제 노트북 실행 결과 기반

---

## 🎓 참고사항
- 모든 코드는 재현 가능하도록 `random_state=42` 고정
- 8개 원천 데이터 소스 병합 및 전처리는 별도 파이프라인 필요 (v1 전처리 과정)
- 본 제출물은 v1 전처리 완료 후부터 시작
