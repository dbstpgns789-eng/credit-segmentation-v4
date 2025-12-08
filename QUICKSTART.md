# 🚀 빠른 시작 가이드

## 📁 폴더 구조 (먼저 확인)

```
final_submission/
│
├── QUICKSTART.md                  # 본 문서 (빠른 시작)
├── README.md                      # 프로젝트 전체 설명
│
├── notebooks/                     # 실행 노트북
│   ├── step1_build_v4_features.ipynb
│   └── step2_train_and_predict.ipynb
│
├── data/                          # 데이터
├── models/                        # 모델 (생성됨)
├── results/                       # 결과 (생성됨)
│
├── project_evolution/             # 프로젝트 문서
│   ├── 01_PROJECT_OVERVIEW.md
│   ├── 02_DATA_UNDERSTANDING.md
│   ├── 03_VERSION_EVOLUTION.md
│   ├── 04_FEATURE_ENGINEERING_STRATEGY.md
│   └── 05_FAILED_EXPERIMENTS.md
│
└── marketing_presentation/        # 마케팅 작업 공간
    ├── README.md
    └── WORK_INSTRUCTIONS.md       # 팀원 작업 지시서
```

---

## 1️⃣ 환경 준비 (1분)

```bash
# 필수 라이브러리 설치
pip install pandas numpy xgboost scikit-learn pyarrow
```

---

## 2️⃣ 데이터 확인 (선택사항)

학습/테스트 데이터가 이미 준비되어 있습니다:

```
data/
├── df_master_v4_train.parquet   ✅ 학습 데이터 (Segment 포함)
└── df_master_v4_test.parquet    ✅ 테스트 데이터
```

**데이터가 없다면**: `notebooks/step1_build_v4_features.ipynb`를 먼저 실행하세요.

---

## 3️⃣ 모델 학습 및 예측 (10-20분)

### 📓 Jupyter Notebook 실행

1. **파일 열기**: `notebooks/step2_train_and_predict.ipynb`

2. **전체 셀 실행**: 
   - Jupyter: `Run All` 버튼 클릭
   - VS Code: 상단 `Run All` 클릭

3. **진행 상황 확인**:
   ```
   ✅ Stage 1 학습 중... (약 5분)
   ✅ Stage 2A 학습 중... (약 2분)
   ✅ Stage 2B 학습 중... (약 5분)
   ✅ Test 예측 중...
   ✅ 저장 완료!
   ```

---

## 4️⃣ 결과 확인

### 📁 생성된 파일

```
results/
├── v4_test_predictions.csv      ⭐ 최종 제출 파일
└── v4_test_predictions.parquet

models/
├── model_stage1_rare_vs_others.pkl
├── model_stage2A_seg01.pkl
├── model_stage2B_seg234.pkl
└── label_encoder_234.pkl
```

### 📊 성능 확인

노트북 출력에서 다음 지표 확인:
- **Stage 1 F1-Score**: rare vs others 분류 성능
- **Stage 2A Macro F1**: Segment 0 vs 1 성능
- **Stage 2B Macro F1**: Segment 2 vs 3 vs 4 성능
- **Overall Macro F1**: 최종 전체 성능 ⭐

---

## 5️⃣ 제출

`results/v4_test_predictions.csv` 파일을 제출하세요!

**파일 형식 확인**:
```csv
ID,Segment_pred,Segment_pred_label
TRAIN_000002,3,D
TRAIN_000007,2,C
...
```

---

## 🔧 문제 해결

### ❌ 메모리 부족 오류
```python
# 노트북 상단에 추가
import gc
gc.collect()
```

### ❌ 파일 경로 오류
프로젝트 루트가 `C:\Users\User\전산통계프로젝트`인지 확인

### ❌ 데이터 파일 없음
`notebooks/step1_build_v4_features.ipynb`를 먼저 실행

---

## 📚 더 자세한 정보

- **전체 문서**: `README.md`
- **프로젝트 구조**: `README.md` 참조
- **피처 설명**: `README.md`의 "v4 피처 엔지니어링 상세" 섹션

---

## ⏱️ 예상 소요 시간

| 단계 | 시간 |
|------|------|
| 환경 설정 | 1분 |
| Step 1 (피처 생성) | 5-10분 |
| Step 2 (모델 학습 및 예측) | 10-20분 |
| **총 소요 시간** | **15-30분** |

---

**🎉 완료! 이제 제출하세요!**
