---
title: SenseDocent 사용자 평가
emoji: 🎨
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# SenseDocent 사용자 평가 (Accessibility HCI Research)

본 저장소는 시각장애인을 위한 AI 공감각 미술 큐레이션 서비스 **SenseDocent** 의
사용자 평가 연구를 위한 Gradio 기반 실험 인터페이스입니다.

---

## 1. 프로젝트 개요

- 30개 명화 × 3개 설명 조건(`caption`, `docent`, `sensedocent`) = **총 90개 음성 자극물**
- 참여자 1명은 1개 블록(A~E)을 배정받고, 그 블록의 6작품 중 **무작위 3작품**(participant_id 기반 고정)을 배정받아 **9개 자극물** (3작품 × 3조건) 을 평가
- 평가 항목
  - 작품 자극물별 **9개 Likert 문항(Q1~Q9)**
  - 자극물별 3개 자유응답(F1~F3)
  - 블록 종료 후 전체 평가 4문항(G1~G4)

## 2. 연구 목적

- AI 기반 비시각 감각 표현이 작품 이해/몰입/사용 의향에 미치는 영향 검증
- 시각장애인을 포함한 사용자가 미술 도슨트 음성을 어떻게 경험하는지 정량/정성 분석

## 3. 파일 구조

```
.
├── app.py                  # Gradio 실행 진입점
├── requirements.txt
├── README.md
├── data/
│   ├── sensedocent_evaluation_stimuli_90_v7_with_audio.csv  # 90개 자극물 정의 (필수)
│   ├── sensedocent_validation_summary_v7_final.csv        # 검증 요약 (선택)
│   └── sensedocent_fact_check_log_v7.csv                  # 팩트체크 로그 (선택, 비공개)
├── audio/                  # *.mp3 평가용 TTS 음성 파일
├── images/                 # <artwork_id>.jpg|png|webp 작품 이미지 (선택)
└── responses/              # 자동 생성. 평가 결과 CSV 저장
    ├── sensedocent_responses.csv
    └── sensedocent_global_feedback.csv
```

`responses/` 폴더는 앱 실행 시 자동으로 생성됩니다.

## 4. 실행 방법 (로컬)

```bash
pip install -r requirements.txt
python app.py
```

브라우저에서 표시된 로컬 URL (`http://127.0.0.1:7860`) 에 접속해 평가를 진행할 수 있습니다.

### 디버그 모드

연구자 확인용으로 condition 명, script_text, 자극물 개수, 누락 오디오 파일 목록을 표시합니다.

```bash
DEBUG_MODE=true python app.py
```

기본값은 `DEBUG_MODE=false` (참여자 모드) 입니다.

## 5. Hugging Face Spaces 배포 방법

1. HF Spaces 에서 새 Space 생성 → SDK `Gradio` 선택
2. 본 저장소를 그대로 push (`app.py`, `requirements.txt`, `README.md`, `data/`, `audio/`)
3. 빌드 완료 후 Space URL 로 접속

> ⚠ Spaces 컨테이너의 파일시스템은 휘발성입니다. **재시작 시 `responses/` 가 소실** 될 수 있습니다.
> 실제 운영 시에는 다음 중 하나를 강력히 권장합니다.
>
> - HF Spaces **Persistent Storage** 활성화
> - HF **Dataset repo** 로 응답 push (예: `huggingface_hub.HfApi().upload_file(...)`)
> - **Google Sheets / Supabase / Notion DB** 등 외부 저장소로 응답 전송

## 6. 데이터 파일 설명

`data/sensedocent_evaluation_stimuli_90_v7_final.csv` 의 주요 컬럼:

| 컬럼 | 의미 |
|------|------|
| `stimulus_id` | 자극물 고유 ID |
| `block_id` | 블록 (A~E) |
| `artwork_id` | 작품 ID |
| `artist_ko` / `title_ko` | 작가명 / 작품명 |
| `condition` | **내부 조건명** (caption / docent / sensedocent) — 참여자에게 노출 금지 |
| `blind_label_for_participant` | 참여자에게 보여줄 라벨 (설명 1/2/3) |
| `script_text` | TTS 원문 (참여자 화면에서는 숨김) |
| `audio_file_suggested` | 매칭될 mp3 파일명 |
| `presentation_order` | 블록 내 제시 순서 |

> 컬럼명이 살짝 달라도 동작하도록 alias 처리되어 있습니다 (예: `artist_ko ↔ artist`).

## 7. 오디오 / 이미지 파일 준비 방법

### 오디오 (`audio/`)

1. CSV 의 `audio_file_suggested` 컬럼 값과 동일한 파일명을 `audio/` 폴더에 저장합니다.
   예: `audio/GOGH_001_caption.mp3`, `audio/GOGH_001_sensedocent.mp3` …
2. mp3 / wav 등 브라우저 재생 가능한 포맷 사용
3. 누락된 파일은 디버그 모드(`DEBUG_MODE=true`)에서 시작 화면에 목록이 표시됩니다.
4. 오디오가 없는 자극물은 평가 화면에 경고가 표시되지만, 진행은 가능합니다
   (실제 연구에서는 모든 음성 파일이 갖춰진 상태에서 운영하세요).

### 이미지 (`images/`, 선택)

이미지 파일 매칭은 두 가지 방식 중 하나로 동작합니다 (자동 감지).

**방식 1. 파일명을 `artwork_id` 와 같게 두기 (가장 단순)**

CSV 의 `artwork_id` 값과 동일한 이름의 이미지 파일을 `images/` 폴더에 저장합니다.
확장자는 `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`, `.tif`, `.tiff` 중 무엇이든 무방.

```
images/GOGH_001.jpg
images/CEZA_001.png
...
```

**방식 2. 매핑 CSV 사용 (Kaggle 등 외부 데이터셋 활용 시)**

`data/image_mapping.csv` 의 `image_file` 컬럼에 실제 파일명(확장자 포함)을 채워주세요.
이 매핑이 위의 `artwork_id` 기반 매칭보다 **우선**합니다.

예 — Kaggle [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)
데이터셋의 파일명을 그대로 쓰는 경우:

```csv
artwork_id,artist_ko,title_ko,image_file
GOGH_001,빈센트 반 고흐,별이 빛나는 밤,Vincent_van_Gogh_42.jpg
MONE_005,클로드 모네,생라자르 역,Claude_Monet_18.jpg
```

이미지 파일들은 매핑에 적힌 파일명 그대로 `images/` 폴더에 넣으면 됩니다
(서브폴더 사용 안 함, `images/` 평면 구조).

**공통 사항**

- 이미지가 없거나 매핑이 비어 있으면 평가 화면에 이미지 영역이 빈 상태로 진행됩니다 (에러 없음).
- 시각장애인/저시력 사용자에게는 이미지가 1차 정보원이 아니므로,
  작품명/작가명/오디오 설명을 통해 평가가 가능하도록 구성되어 있습니다.
- alt 텍스트는 자동으로 "작품 이미지: 작가명 - 작품명" 형식으로 채워집니다 (스크린리더 호환).
- `DEBUG_MODE=true` 로 어떤 이미지가 누락됐는지, 매핑이 몇 개 로드됐는지 확인할 수 있습니다.

## 8. 응답 저장 방식

### `responses/sensedocent_responses.csv` (자극물 단위)

저장 컬럼:
```
participant_id, participant_group, block_id, timestamp,
stimulus_index, stimulus_id, artwork_id, artist, title,
condition, blind_label_for_participant, audio_file,
q1 ~ q9,
good_expression, awkward_expression, improvement_comment,
audio_play_count, response_start_time, response_end_time, response_time_sec
```

- 매 자극물 응답 제출 시 **append** 방식으로 즉시 저장됩니다.
- 분석을 위해 `condition` 컬럼은 저장하지만, **화면에는 노출하지 않습니다.**

### `responses/sensedocent_global_feedback.csv` (참여자 단위)

저장 컬럼:
```
participant_id, participant_group, block_id, timestamp,
g1_best_imagery, g2_best_immersion, g3_best_service_use, g4_overall_comment
```

### Hugging Face Dataset repo 자동 백업 (권장)

HF Space 컨테이너의 `responses/` 폴더는 휘발성입니다 (재시작/재배포 시 소실).
다음 두 환경변수를 설정하면 매 응답이 **사설 HF Dataset repo 에 개별 JSON 파일**로 자동 push 됩니다.

| 변수 | 종류 | 예시 |
|------|------|------|
| `HF_TOKEN` | **Secret** (Space Settings → Variables and secrets) | `hf_xxxxxxxxxxxx` (write 권한) |
| `HF_DATASET_REPO` | **Variable** | `s00hyuk/sensedocent-responses` |

**셋업 절차**

1. https://huggingface.co/new-dataset 에서 새 dataset repo 생성
   - Owner: 본인 / Name: `sensedocent-responses` 등 / **Private** 체크
2. https://huggingface.co/settings/tokens 에서 **write 권한 토큰** 생성
3. Space 페이지 → **Settings** → **Variables and secrets**
   - `HF_TOKEN` 을 **Secret** 으로 등록 (위 토큰 값)
   - `HF_DATASET_REPO` 를 **Variable** 으로 등록 (`사용자명/repo이름`)
4. Space 재시작 (자동) 후 동작

**Dataset 파일 구조**

```
sensedocent-responses/
├── stimulus/
│   └── <participant_id>/
│       └── <participant_id>__<stimulus_id>__<timestamp>.json   # 각 자극물 응답
└── global/
    └── <participant_id>__<timestamp>.json                       # 블록 종료 후 응답
```

각 응답이 별도 파일이라 동시 참여자가 있어도 race condition 없음.
업로드는 백그라운드 스레드에서 처리되어 평가 응답성에 영향을 주지 않습니다.
업로드 실패해도 로컬 CSV 에는 정상 저장되므로 데이터 유실 위험은 이중으로 막아둔 상태입니다.

**분석 시 활용**

본인 PC 에서:
```bash
git clone https://huggingface.co/datasets/<user>/sensedocent-responses
cd sensedocent-responses
# JSON 들을 pandas DataFrame 으로
python3 -c "
import json, glob, pandas as pd
rows = [json.load(open(p, encoding='utf-8')) for p in glob.glob('stimulus/**/*.json', recursive=True)]
df = pd.DataFrame(rows)
print(df.head()); print(df.shape)
"
```

## 9. 주의사항

- **condition 은 참여자에게 절대 노출하지 마세요.** 화면에는 `blind_label_for_participant`
  ("설명 1/2/3") 만 표시됩니다.
- 실제 연구 운영 전 반드시 다음을 확인하세요.
  - 모든 음성 mp3 파일이 `audio/` 폴더에 존재하는지
  - CSV 의 `audio_file_suggested` 값과 실제 파일명이 일치하는지
  - 평가 흐름이 모바일 / 키보드 / 스크린리더 환경에서도 동작하는지
- 응답 저장은 컨테이너 휘발성을 고려하여 **Persistent Storage 또는 외부 저장소** 연동을 권장합니다.
- 참여자가 동의 체크박스를 선택하지 않으면 평가가 시작되지 않습니다.
- 9개 Likert 문항(Q1~Q9) 은 필수 응답, 자유응답(F1~F3) 은 선택 입력입니다.

## 10. 접근성 고려

- 큰 글씨 / 고대비 색상 / 큰 버튼 / 단순한 카드형 레이아웃
- 키보드 탐색 가능 / 스크린리더 친화적 라벨
- 한 화면에 하나의 자극물만 표시 / 진행률 명시
- 모바일 반응형 (Gradio Blocks + 미디어 쿼리)

## 11. 라이선스

연구 목적의 평가 도구. 평가 자극물(음성/이미지)의 저작권은 원저작자에게 귀속됩니다.
