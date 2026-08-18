# 2026 여름 산학협력 실습중심 딥러닝 부트캠프 — 6조 eclipse

> 🏆 **서울시립대학교 인공지능혁신융합대학사업단장상 수상**

MLLM(Qwen3-VL-8B-Instruct) 기반 **Visual Chain-of-Thought** 전략으로 두 개의 경진대회 과제를 푼 최종 제출 코드입니다.
두 과제 모두 학습(fine-tuning) 없이, **서버에 제공된 사전학습 모델 + 프롬프트 설계 + 파이프라인 구성**만으로 성능을 끌어올렸습니다.

| 과제 | 태스크 | 평가지표 | 배점 | 최종 점수 |
|---|---|---|---|---|
| Problem 1 | 상권 이미지 업종 분류 (VQA, 8지선다) | Accuracy | 20% | **0.765** |
| Problem 2 | Reasoning Segmentation (이미지 + 암시적 쿼리) | Dice Score | 40% | **0.50968** |

**팀원** — 이유진(성균관대 컴퓨터교육과) · 박종혁(서울시립대 컴퓨터과학부) · 강민성(서울과학기술대 인공지능응용학과)

**수상** — 서울시립대학교 인공지능혁신융합대학사업단장상 (2026.08)

---

## 저장소 구조

```
.
├── 6조 eclipse.pdf                                 # 최종 발표 자료 (문제 정의 · 실험 로그 · 시행착오)
├── problem1/
│   ├── ch1_1_eclipse.ipynb                         # 업종 분류 파이프라인 (실행 출력 포함)
│   └── sample_submission.csv                       # 제출 양식 (ID, label / 200행)
└── problem2/
    ├── 01_assignment2_baseline_0.49385.ipynb       # ① baseline: Qwen3-VL + SAM2.1
    ├── 03_assignment2_multi_specialist.ipynb       # ② multi: 복수 타깃 특화 프롬프트
    ├── assignment2_ensemble_base_multi_only.ipynb  # ③ ①·② 마스크 OR 앙상블 (최종 제출본)
    └── sample_submission2.csv                      # 제출 양식 (ID, label / 200행)
```

> Problem 2는 **① → ② → ③ 순서로 반드시 실행**해야 합니다. ③은 ①·②가 생성한 CSV 두 개를 입력으로 받습니다.

---

## 공통 환경 및 제약

- 커널: `Bootcamp Environment 1 (TGM Shared)` (부트캠프 제공 GPU 서버, Jupyter)
- 주요 의존성: `torch`, `transformers`(Qwen3-VL), `sam2`, `torchvision`, `pandas`, `numpy`, `Pillow`, `tqdm`
- 대회 규정 준수 사항
  - 서버에 제공된 **Qwen3-VL-8B-Instruct** / **SAM2.1 Hiera-Large** 만 사용
  - 외부 API · 외부 데이터 · 외부 모델 다운로드 없음
  - `SEED = 42` 고정, 디코딩은 `do_sample=False`(greedy)로 **결정론적 추론**
  - Problem 2는 대회 안내에 따라 `torchvision.datasets.ImageFolder()` 로 테스트셋 로드
  - `sample_submission.csv` 의 행 순서·컬럼을 그대로 유지
  - 추론 실패를 임의의 빈 마스크로 대체하지 않음 (실패 시 저장 단계에서 예외 발생)

---

## Problem 1 — 상권 이미지 업종 분류

### 문제 정의
1000×1000 RGB 상권 이미지(Test 200장)를 보고 매장의 업종을 8지선다로 분류합니다.
정답은 **알파벳 한 글자**만 인정됩니다.

`A. 한식 · B. 양식 · C. 일식 · D. 중식 · E. 카페 · F. 베이커리 · G. 주점 · H. 동남아`

### 모델 선정 — Qwen3-VL-8B-Instruct
- **Bounding box 좌표를 직접 출력**할 수 있어, 간판·메뉴판 등 판단 근거 영역을 모델이 스스로 지목(grounding)할 수 있음
- 한국어 텍스트와 이미지를 동시에 이해 → 간판/메뉴의 한글 판독이 결정적인 본 과제에 적합
  (실제로 분류를 CLIP에 전담시킨 실험은 한국어 미이해로 **한식 편향**이 발생해 폐기)

### 파이프라인 — 2-Turn Visual CoT

```text
원본 이미지
   ↓  Turn 1 : Qwen3-VL
   판단 근거가 되는 영역 BBox 3개 탐색 (0~1000 정규화 좌표)
   ↓  crop (padding 10%, 최소 224×224 보정)
   ↓  Turn 2 : Qwen3-VL
   원본 + crop 3장 + CoT 유도 프롬프트 → "Answer: X"
   ↓  파싱 실패 시 fallback
   원본 1장 + 1토큰 강제 출력 프롬프트
   ↓
submission.csv (ID, label)
```

세 개의 프롬프트가 각 단계를 담당합니다.

| 프롬프트 | 역할 | 설계 포인트 |
|---|---|---|
| `TURN1_PROMPT` | 근거 영역 3개 탐색 | 정확히 3개 강제, **간판·메뉴판·매장 전면 디자인 등 유의미한 단서를 예시로 제시**해 탐색 방향 유도, 배경(하늘·행인·옆 가게) 배제 |
| `TURN2_PROMPT` | 최종 분류 | 근거 우선순위 명시(① 판독 가능한 텍스트 ② 음식·상품 ③ 인테리어·분위기), 근거 1~2개를 짚는 **step-by-step 추론 유도**, `Answer: X` 형식 강제 |
| `FINAL_PROMPT` | 파싱 실패 fallback | 원본만 보고 **정확히 한 글자**만 출력 (`FINAL_MAX_NEW_TOKENS = 1`) |

출력 파싱은 Qwen native 박스 토큰(`<|box_start|>…<|box_end|>`)을 우선 시도하고, 실패 시 `[x1, y1, x2, y2]` 형태를 정규식으로 회수합니다. 라벨은 `Answer:\s*([A-H])` 의 **마지막 매치**를 취합니다.

### 주요 설정
| 항목 | 값 |
|---|---|
| `RESIZE_SIZE` | 840 (모든 입력 이미지 정사각 리사이즈) |
| `N_CROPS` | 3 |
| crop padding | 0.10 (최소 224×224 보장) |
| max_new_tokens | Turn1 256 / Turn2 512 / Final 1 |
| `SEED` | 42 |

### 실행 방법
1. `ch1_1_eclipse.ipynb` 첫 셀의 `MODEL_PATH`, `IMAGE_DIR` 을 채점 환경 경로로 수정
2. 노트북을 위에서부터 순차 실행
3. 작업 폴더에 `submission.csv` 생성

### 실행 결과 (노트북에 저장된 로그)
- 200장 추론 **32분 17초** (약 9.7초/장, 단일 GPU)
- BBox가 3개 미만이었던 이미지: **0 / 200**
- Turn2 파싱 실패로 fallback한 이미지: **1**, fallback에서도 실패: **0**
- 예측 라벨 분포: A 48 · B 26 · C 23 · D 21 · E 32 · F 16 · G 20 · H 14 (결측 0, 형식 위반 0)

### 시행착오 및 개선 (발표자료 기준)
| 실험 | 변경 내용 | Accuracy | 판단 |
|---|---|---|---|
| Baseline 변경 | 단일 MLLM 분류 → **MLLM crop + MLLM 분류** 2단계 | 0.735 | 2단계 파이프라인 채택 |
| Crop 프롬프트 개선 | 간판·메뉴판 등 **단서 예시 제공**으로 BBox 탐색 유도 | 0.750 | 기존에 crop 못 하던 이미지도 탐색 성공 |
| 최대 crop 수 3 → 2 | 환각 억제 목적 | 0.745 | 성능 하락 → **3 유지** |
| 분류 프롬프트 개선 | **CoT 유도 + 관찰 포인트 예시 제공** | **0.765** | 최종 채택 |
| Baseline 변경 2 | 분류를 CLIP에 전담 | 한식 편향 | 한국어 미이해로 폐기 |

---

## Problem 2 — Reasoning Segmentation

### 문제 정의
이미지와 **암시적·복합적 텍스트 쿼리**(예: *"공격에 사용되는 신체 부위 중, 먹잇감을 빠르게 무력화시키는 데 사용되는 부위는 무엇인가요?"*)를 입력받아, 단순 객체 탐지를 넘어 문맥 추론으로 대상 영역을 분할합니다.
1000×1000 RGB, Test 200장, 평가지표는 **Dice Score**.

### 접근 — Problem 1 파이프라인 + SAM2.1
Problem 1에서 검증된 2-Turn Visual CoT 구조를 그대로 옮기고, 마지막에 **SAM2.1 Hiera-Large** 를 붙여 좌표를 마스크로 변환했습니다.
Qwen3-VL은 **어디를(bbox + positive point)**, SAM은 **정확한 경계**를 담당하는 역할 분리 구조입니다.
(SAM3로 교체한 실험은 0.47719로 SAM2.1과 유의미한 차이가 없어 SAM2.1 유지)

```text
원본 이미지 + 쿼리
   ↓  Turn 1 : Qwen3-VL
   쿼리 해결에 유용한 영역 BBox 최대 3개 (타깃 / 참조 객체 / 구별 속성 / 공간 관계)
   ↓  crop / zoom
   ↓  Turn 2 : Qwen3-VL (원본 + crop + 쿼리)
   최종 target bbox + positive point → <answer> JSON 블록
   ↓  파싱 실패 시 fallback : 원본만으로 직접 localization
   ↓  SAM2.1 (bbox + point 프롬프트, multimask_output → score 최대 마스크 선택)
   타깃이 여러 개면 마스크 logical OR
   ↓  1000×1000 bool 로 리사이즈 (NEAREST)
   ↓  RLE (mask.T.flatten() 기준, 대회 제공 방식과 동일)
submission.csv
```

### 노트북별 역할

| 노트북 | 산출물 | 설명 |
|---|---|---|
| `01_assignment2_baseline_0.49385.ipynb` | `submission_task2_baseline.csv`<br>`debug_task2_baseline.csv` | 기본 파이프라인. 단일 타깃 중심으로 안정적인 마스크 생성 |
| `03_assignment2_multi_specialist.ipynb` | `submission_task2_multi_specialist.csv`<br>`debug_task2_multi_specialist.csv` | **Turn2 프롬프트에 복수 인스턴스 지시문 2줄만 추가**한 변형 |
| `assignment2_ensemble_base_multi_only.ipynb` | `submission_task2_ensemble_base_multi.csv` | 두 CSV를 마스크로 복원 → **픽셀 단위 OR** → 재인코딩 (**최종 제출본**) |

`01` 과 `03` 의 코드 차이는 출력 경로와 아래 지시문 두 줄이 전부입니다.

```text
Before finalizing, carefully determine from the original query whether the intended answer
contains one target or multiple distinct target instances.
When multiple distinct candidates independently satisfy all query conditions,
return a separate annotation for each matching target.
```

### 왜 앙상블인가
- **기본 프롬프트의 문제** — 조건을 만족하는 후보가 여럿일 때 하나만 선택하고, crop 중 한 장에만 의존하는 현상 발생 → 복수 정답 지시문 추가(`multi`)
- **multi의 문제** — 복수 정답은 잘 잡지만, 기존 baseline이 맞히던 것을 놓치는 **hallucination** 발생 (단독 점수는 0.48로 오히려 하락)
- **해결** — 두 모델은 서로 다른 역할을 하므로, baseline이 맞힌 영역은 그대로 가져가고 multi가 추가로 찾은 복수 정답을 마스크에 더하는 **OR 앙상블**. multi의 hallucination 손실보다 복수 정답으로 얻는 Dice 이득이 크다고 판단 → **0.50968 (최고 점수)**

### 견고성 장치
- **이미지 ↔ 쿼리 매칭**: `query.csv` ID → `sample_submission` ID → natural sort 순으로 폴백, 컬럼명도 후보 리스트 기반 자동 탐색(`pick_column`)
- **flat `imgs/` 대응**: `ImageFolder` 로 직접 못 읽으면 `test_class/` 하위에 symlink 뷰를 만들어 규정 충족
- **좌표 보정**: bbox 정렬·clamp, positive point가 bbox 밖이면 중심점으로 대체, `bbox_2d` 파싱 실패 시 본문 정규식으로 회수
- **디버그 기록**: 10행마다 Turn1/Turn2 원문, bbox, point, SAM score, 마스크 면적비, 오류를 CSV로 체크포인트 저장
- **저장 가드**: 전체 200행을 실행하지 않았거나 실패 행이 하나라도 있으면 `RuntimeError` — 빈 마스크로 은폐하지 않음
- **SAM2 config 주의**: 서버의 `build_sam2()` 는 Hydra config search path를 사용하므로 checkpoint는 **절대경로**, config는 `configs/sam2.1/sam2.1_hiera_l.yaml` **이름**으로 전달해야 합니다 (yaml 절대경로를 넘기면 `MissingConfigException`)

### 시행착오 및 개선 (발표자료 기준)
| 유형 | 변경 내용 | Dice | 판단 |
|---|---|---|---|
| Baseline | 사전학습 모델 + 기본 설정 | 0.498 | Problem 1 파이프라인에 SAM을 붙이는 전략이 매우 효율적 |
| 프롬프트 변경 1 | multi (복수 정답 지시) | 0.48 | 복수 정답 마스킹 능력은 확인, 단독 성능은 하락 |
| 프롬프트 변경 2 | box 위주 탐색 강화 | 0.477 | hallucination 크게 증가 |
| 모델 변경 | SAM3로 교체 | 0.47719 | SAM2.1과 유의미한 차이 없음 |
| **앙상블** | **baseline + multi (OR)** | **0.50968** | **최종 채택** |

> 노트북 파일명의 `0.49385` 는 해당 baseline 실행본의 점수이며, 발표자료의 0.498은 별도 제출 회차 기준입니다.

---

## 재현 가이드

두 과제 모두 경로가 하드코딩되어 있어, **첫 설정 셀만 채점 환경에 맞게 수정**하면 됩니다.

| 상수 | 위치 | 현재 값 |
|---|---|---|
| `MODEL_PATH` | 공통 | `.../models/Qwen3-VL-8B-Instruct` |
| `IMAGE_DIR` | P1 | `.../competitions/competition1/images` |
| `SAM2_MODEL_PATH` | P2 | `.../models/SAM2.1/weights/sam2.1_hiera_large.pt` |
| `SAM2_CFG_NAME` | P2 | `configs/sam2.1/sam2.1_hiera_l.yaml` (Hydra 이름, 경로 아님) |
| `BASE_DIR` | P2 | `.../competitions/competition2` (하위 `imgs/`, `query.csv`) |
| `SAMPLE_SUB_CSV` | 공통 | 노트북과 같은 폴더의 샘플 제출 CSV |

**제출 파일 규격**
- Problem 1: `ID, label` 200행, label은 `A`~`H` 한 글자
- Problem 2: `ID, label` 200행, label은 1000×1000 bool 마스크를 `mask.T.flatten()` 기준으로 인코딩한 RLE 문자열

**부분 실행 옵션 (Problem 2)** — 디버깅 시 `START_INDEX` / `END_INDEX` 로 구간 실행이 가능하지만, 최종 제출 CSV 생성 셀은 `0` / `None` 전체 실행이 아니면 실행을 거부합니다.

---

## 알려진 이슈

- **경로 하드코딩** — 부트캠프 서버(`/home/jjs2403/...`) 기준입니다. 다른 환경에서는 위 표의 상수를 반드시 수정해야 합니다.
- **`problem1/ch1_1_eclipse.ipynb` 의 `n_bbox_short`** — 추론 루프에서 초기화 없이 `n_bbox_short += 1` 을 수행합니다. BBox가 3개 미만인 이미지가 나오면 `NameError` 가 발생하므로, 재실행 전 루프 앞에 `n_bbox_short = 0` 을 추가하는 것이 안전합니다. (저장된 실행 로그에서는 해당 케이스가 0건이라 문제가 드러나지 않았습니다.)
- **Problem 2 노트북에는 실행 출력이 저장되어 있지 않습니다.** 각 실행의 상세 근거는 실행 시 생성되는 `debug_task2_*.csv` 에 기록됩니다.
- **`SHOW_EACH` / `SHOW_CROPS`** 가 `True` 이면 첫 이미지의 시각화가 출력되고, 매 이미지의 Turn1/Turn2 원문이 stdout으로 출력되어 로그가 길어집니다.
