"""
SenseDocent 사용자 평가 연구용 Gradio 앱.

본 앱은 시각장애인을 포함한 사용자가 AI 기반 명화 오디오 큐레이션을
어떻게 경험하는지 평가하기 위한 실험 도구입니다.

중요:
- condition (caption / docent / sensedocent) 값은 참여자 화면에 절대 노출하지 않습니다.
- 참여자에게는 blind_label_for_participant ("설명 1/2/3") 만 노출합니다.
- 응답 CSV에는 분석을 위해 condition 정보를 저장합니다.
- Hugging Face Spaces 의 컨테이너 파일시스템은 재시작 시 초기화될 수 있습니다.
  실제 운영 시에는 HF Persistent Storage, HF Dataset repo, Google Sheets, Supabase 등의
  외부/영구 저장소 연동을 권장합니다.
"""

from __future__ import annotations

import csv
import json
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd

# --------------------------------------------------------------------------- #
# gradio_client monkey-patch
# gr.State(dict) 가 만드는 {additionalProperties: True} 스키마에서
# gradio_client.utils 가 bool 입력을 못 다루는 알려진 버그를 우회한다.
# (TypeError: argument of type 'bool' is not iterable)
# --------------------------------------------------------------------------- #
try:
    import gradio_client.utils as _gc_utils

    _orig_get_type = _gc_utils.get_type
    _orig_jsts = _gc_utils._json_schema_to_python_type

    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        return _orig_get_type(schema)

    def _safe_jsts(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_jsts(schema, defs)

    _gc_utils.get_type = _safe_get_type
    _gc_utils._json_schema_to_python_type = _safe_jsts
except Exception:
    pass

# --------------------------------------------------------------------------- #
# 환경 / 경로 설정
# --------------------------------------------------------------------------- #

DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = BASE_DIR / "audio"
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR_ALT = BASE_DIR / "image"  # singular fallback
IMAGE_SEARCH_DIRS = [IMAGES_DIR, IMAGES_DIR_ALT]
RESPONSES_DIR = BASE_DIR / "responses"
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

STIMULI_CSV_CANDIDATES = [
    DATA_DIR / "sensedocent_evaluation_stimuli_90_v7_with_audio.csv",
    DATA_DIR / "sensedocent_evaluation_stimuli_90_v7_final.csv",
]
STIMULI_CSV = next((p for p in STIMULI_CSV_CANDIDATES if p.exists()), STIMULI_CSV_CANDIDATES[0])
RESPONSES_CSV = RESPONSES_DIR / "sensedocent_responses.csv"
GLOBAL_CSV = RESPONSES_DIR / "sensedocent_global_feedback.csv"

# --------------------------------------------------------------------------- #
# 컬럼 이름 호환 (CSV 컬럼명이 살짝 달라도 동작하도록)
# --------------------------------------------------------------------------- #

COLUMN_ALIASES = {
    "stimulus_id": ["stimulus_id"],
    "artwork_id": ["artwork_id"],
    "artist": ["artist", "artist_ko"],
    "title": ["title", "title_ko"],
    "artwork_year": ["artwork_year", "year"],
    "condition": ["condition"],
    "blind_label_for_participant": ["blind_label_for_participant", "blind_label"],
    "script_text": ["script_text"],
    "audio_file": ["audio_path", "audio_file_suggested", "audio_file"],
    "block_id": ["block_id"],
    "presentation_order": ["presentation_order"],
}


def resolve_column(df: pd.DataFrame, key: str) -> Optional[str]:
    """CSV에 존재하는 실제 컬럼명을 반환. 없으면 None."""
    for cand in COLUMN_ALIASES.get(key, [key]):
        if cand in df.columns:
            return cand
    return None


# --------------------------------------------------------------------------- #
# CSV 로딩
# --------------------------------------------------------------------------- #


def load_stimuli() -> pd.DataFrame:
    if not STIMULI_CSV.exists():
        names = ", ".join(p.name for p in STIMULI_CSV_CANDIDATES)
        raise FileNotFoundError(
            f"자극물 CSV가 존재하지 않습니다.\n"
            f"data/ 폴더에 다음 중 하나의 파일을 두세요: {names}"
        )

    df = pd.read_csv(STIMULI_CSV, encoding="utf-8-sig")
    # BOM 정리
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]

    # 필수 컬럼 점검
    required = ["stimulus_id", "artwork_id", "condition", "block_id"]
    for r in required:
        if resolve_column(df, r) is None:
            raise ValueError(f"필수 컬럼 누락: {r}")

    return df


try:
    STIMULI_DF = load_stimuli()
    LOAD_ERROR: Optional[str] = None
except Exception as e:  # 데이터 로딩 에러는 UI에서 명확히 표시
    STIMULI_DF = pd.DataFrame()
    LOAD_ERROR = str(e)


# --------------------------------------------------------------------------- #
# 자극물 셔플/정렬
# --------------------------------------------------------------------------- #


def get_block_stimuli(block_id: str) -> list[dict]:
    """주어진 block_id의 자극물 18개를 반환.

    presentation_order가 있으면 그 순서, 없으면 (artwork_id, blind_label)으로 정렬.
    """
    if STIMULI_DF.empty:
        return []

    block_col = resolve_column(STIMULI_DF, "block_id")
    sub = STIMULI_DF[STIMULI_DF[block_col].astype(str) == str(block_id)].copy()
    if sub.empty:
        return []

    order_col = resolve_column(sub, "presentation_order")
    artwork_col = resolve_column(sub, "artwork_id")
    blind_col = resolve_column(sub, "blind_label_for_participant")

    if order_col and sub[order_col].notna().all():
        sub = sub.sort_values(by=[artwork_col, order_col])
    elif blind_col:
        sub = sub.sort_values(by=[artwork_col, blind_col])
    else:
        sub = sub.sort_values(by=[artwork_col])

    stimuli = []
    for _, row in sub.iterrows():
        stim = {
            "stimulus_id": str(row.get(resolve_column(sub, "stimulus_id"), "")),
            "artwork_id": str(row.get(artwork_col, "")),
            "artist": str(row.get(resolve_column(sub, "artist"), "")) if resolve_column(sub, "artist") else "",
            "title": str(row.get(resolve_column(sub, "title"), "")) if resolve_column(sub, "title") else "",
            "artwork_year": (
                str(row.get(resolve_column(sub, "artwork_year"), ""))
                if resolve_column(sub, "artwork_year")
                else ""
            ),
            "condition": str(row.get(resolve_column(sub, "condition"), "")),
            "blind_label": str(row.get(blind_col, "")) if blind_col else "",
            "script_text": (
                str(row.get(resolve_column(sub, "script_text"), ""))
                if resolve_column(sub, "script_text")
                else ""
            ),
            "audio_file": (
                str(row.get(resolve_column(sub, "audio_file"), ""))
                if resolve_column(sub, "audio_file")
                else ""
            ),
            "block_id": str(row.get(block_col, "")),
        }
        stimuli.append(stim)
    return stimuli


N_ARTWORKS_PER_PARTICIPANT = 3


def select_artworks_for_participant(
    stimuli: list[dict], participant_id: str, n: int = N_ARTWORKS_PER_PARTICIPANT
) -> list[dict]:
    """블록의 작품들 중 participant_id 기반으로 결정적(deterministic) 무작위 n개 선택.

    - 같은 participant_id 는 항상 같은 n개 작품을 받음 (재접속/새로고침에도 동일).
    - 선택된 작품의 자극물들은 원래 정렬 순서(작품별 3개 조건)를 유지.
    - 작품이 n개 이하이면 전부 반환.
    """
    if not stimuli:
        return []

    # 작품 ID 를 등장 순서대로 (중복 제거)
    artwork_ids: list[str] = []
    for s in stimuli:
        aid = s.get("artwork_id", "")
        if aid and aid not in artwork_ids:
            artwork_ids.append(aid)

    if len(artwork_ids) <= n:
        return stimuli

    rng = random.Random(str(participant_id))
    chosen = set(rng.sample(artwork_ids, n))
    # 원래 순서 유지하며 선택된 작품의 자극물만 필터
    return [s for s in stimuli if s.get("artwork_id", "") in chosen]


def resolve_audio_path(filename: str) -> Optional[str]:
    if not filename or filename.lower() == "nan":
        return None
    # 절대 경로일 수도 있고, 단순 파일명일 수도 있음
    p = Path(filename)
    if p.is_absolute() and p.exists():
        return str(p)
    candidate = AUDIO_DIR / p.name
    if candidate.exists():
        return str(candidate)
    return None


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"]


def _load_image_mapping() -> dict[str, str]:
    """data/image_mapping.csv 가 있으면 {artwork_id: image_file} 로 읽어온다.

    형식 (헤더 필수):
        artwork_id,image_file
        GOGH_001,Vincent_van_Gogh_42.jpg
        MONE_005,Claude_Monet_18.jpg
    """
    mapping_csv = DATA_DIR / "image_mapping.csv"
    if not mapping_csv.exists():
        return {}
    try:
        df = pd.read_csv(mapping_csv, encoding="utf-8-sig")
        df.columns = [c.strip().lstrip("﻿") for c in df.columns]
        if "artwork_id" not in df.columns or "image_file" not in df.columns:
            return {}
        out = {}
        for _, row in df.iterrows():
            aid = str(row.get("artwork_id", "")).strip()
            f = str(row.get("image_file", "")).strip()
            if aid and f and f.lower() != "nan":
                out[aid] = f
        return out
    except Exception:
        return {}


IMAGE_MAPPING = _load_image_mapping()


def resolve_image_path(artwork_id: str) -> Optional[str]:
    """이미지 파일 매칭 순서.

    1. data/image_mapping.csv 에 명시된 파일명 (확장자 포함) 그대로 시도
    2. images/<artwork_id>.<ext>  (확장자는 IMAGE_EXTENSIONS 순서대로)

    탐색 폴더는 images/ (복수) 와 image/ (단수) 둘 다 시도.
    """
    if not artwork_id or artwork_id.lower() == "nan":
        return None

    # 1) 매핑 CSV 우선
    mapped = IMAGE_MAPPING.get(artwork_id) or IMAGE_MAPPING.get(str(artwork_id).strip())
    if mapped:
        for d in IMAGE_SEARCH_DIRS:
            # 확장자 포함된 파일명 그대로
            candidate = d / Path(mapped).name
            if candidate.exists():
                return str(candidate)
            # 확장자가 빠진 경우 보조 시도
            stem = Path(mapped).stem
            for ext in IMAGE_EXTENSIONS:
                for variant in (ext, ext.upper()):
                    c = d / f"{stem}{variant}"
                    if c.exists():
                        return str(c)

    # 2) artwork_id 그대로
    for d in IMAGE_SEARCH_DIRS:
        for ext in IMAGE_EXTENSIONS:
            for variant in (ext, ext.upper()):
                candidate = d / f"{artwork_id}{variant}"
                if candidate.exists():
                    return str(candidate)
    return None


# --------------------------------------------------------------------------- #
# 응답 저장
# --------------------------------------------------------------------------- #

RESPONSE_FIELDS = [
    "participant_id",
    "participant_group",
    "block_id",
    "timestamp",
    "stimulus_index",
    "stimulus_id",
    "artwork_id",
    "artist",
    "title",
    "condition",
    "blind_label_for_participant",
    "audio_file",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q7",
    "q8",
    "q9",
    "good_expression",
    "awkward_expression",
    "improvement_comment",
    "audio_play_count",
    "response_start_time",
    "response_end_time",
    "response_time_sec",
]

GLOBAL_FIELDS = [
    "participant_id",
    "participant_group",
    "block_id",
    "timestamp",
    "g1_best_imagery",
    "g2_best_immersion",
    "g3_best_service_use",
    "g4_overall_comment",
]


def append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    is_new = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        # 누락된 키는 빈 문자열로
        safe_row = {k: row.get(k, "") for k in fieldnames}
        writer.writerow(safe_row)


# --------------------------------------------------------------------------- #
# Hugging Face Dataset repo 자동 백업
#
# Space 컨테이너의 /app/responses 는 휘발성이므로, 매 응답마다 별도 JSON 파일을
# 사설(private) HF Dataset repo 에 push 한다. 파일이 응답마다 분리되므로 동시
# 참여자 간 race condition 없음.
#
# 필요한 환경 변수 (HF Space → Settings → Variables and secrets):
#   HF_TOKEN          (secret)   write 권한 access token (hf_xxxxx)
#   HF_DATASET_REPO   (variable) ex: "s00hyuk/sensedocent-responses"
#
# 둘 중 하나라도 비어 있으면 자동 백업은 비활성화되고, 로컬 CSV 만 기록된다.
# --------------------------------------------------------------------------- #

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "").strip()

_HF_API: Optional[object] = None
_HF_API_LOCK = threading.Lock()
_HF_REPO_READY = False


def _get_hf_api():
    """huggingface_hub.HfApi 를 lazy-init 으로 반환. 환경변수 미설정 시 None."""
    global _HF_API, _HF_REPO_READY
    if not (HF_TOKEN and HF_DATASET_REPO):
        return None
    if _HF_API is not None:
        return _HF_API
    with _HF_API_LOCK:
        if _HF_API is not None:
            return _HF_API
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            # repo 존재 확인 + private dataset 으로 생성 시도 (없으면)
            try:
                api.create_repo(
                    repo_id=HF_DATASET_REPO,
                    repo_type="dataset",
                    private=True,
                    exist_ok=True,
                )
                _HF_REPO_READY = True
            except Exception as e:
                print(f"[hub-backup] create_repo skipped: {e}")
                _HF_REPO_READY = True  # 이미 있을 가능성 — upload 시도해본다
            _HF_API = api
            print(f"[hub-backup] enabled → {HF_DATASET_REPO}")
            return _HF_API
        except Exception as e:
            print(f"[hub-backup] init failed: {e}")
            return None


def _safe_token(s: str) -> str:
    """파일명에 쓸 수 있도록 안전한 토큰화."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in (s or ""))


def hub_push_response(row: dict, kind: str) -> None:
    """응답 한 건을 HF Dataset repo 에 개별 JSON 파일로 업로드.

    kind:
      - 'stimulus' → stimulus/<participant>/<participant>_<stimulus_id>_<ts>.json
      - 'global'   → global/<participant>_<ts>.json

    실패해도 평가 흐름은 막지 않는다 (로컬 CSV 에는 이미 저장됨).
    """
    api = _get_hf_api()
    if api is None:
        return

    pid = _safe_token(str(row.get("participant_id", "anon")))
    ts = _safe_token(str(row.get("timestamp", datetime.now().isoformat(timespec="seconds"))))

    if kind == "stimulus":
        stim_id = _safe_token(str(row.get("stimulus_id", "x")))
        path_in_repo = f"stimulus/{pid}/{pid}__{stim_id}__{ts}.json"
        commit_msg = f"stimulus: {pid} / {stim_id}"
    else:
        path_in_repo = f"global/{pid}__{ts}.json"
        commit_msg = f"global: {pid}"

    payload = json.dumps(row, ensure_ascii=False, indent=2).encode("utf-8")

    def _do_upload():
        try:
            api.upload_file(
                path_or_fileobj=payload,
                path_in_repo=path_in_repo,
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                commit_message=commit_msg,
            )
        except Exception as e:
            print(f"[hub-backup] upload failed ({path_in_repo}): {e}")

    # 응답 제출 응답성을 떨어뜨리지 않도록 백그라운드 스레드로 업로드
    threading.Thread(target=_do_upload, daemon=True).start()


# --------------------------------------------------------------------------- #
# 세션 상태
# --------------------------------------------------------------------------- #


@dataclass
class SessionState:
    participant_id: str = ""
    participant_group: str = ""
    block_id: str = ""
    stimuli: list[dict] = field(default_factory=list)
    current_index: int = 0
    audio_play_count: int = 0
    response_start_time: float = 0.0

    def total(self) -> int:
        return len(self.stimuli)

    def current(self) -> Optional[dict]:
        if 0 <= self.current_index < len(self.stimuli):
            return self.stimuli[self.current_index]
        return None


def empty_state() -> dict:
    return SessionState().__dict__.copy()


def state_from_dict(d: dict) -> SessionState:
    return SessionState(**d)


# --------------------------------------------------------------------------- #
# Likert 옵션
# --------------------------------------------------------------------------- #

LIKERT_CHOICES = [
    ("1 - 매우 아니다", 1),
    ("2 - 아니다", 2),
    ("3 - 보통이다", 3),
    ("4 - 그렇다", 4),
    ("5 - 매우 그렇다", 5),
]

LIKERT_QUESTIONS = [
    ("q1", "Q1. 설명을 통해 작품 속 대상들의 상대적 위치와 배치를 머릿속에 떠올리기 쉬웠다."),
    ("q2", "Q2. 설명을 들은 뒤 작품의 전체적인 공간 구조가 비교적 선명하게 상상되었다."),
    ("q3", "Q3. 설명 속 감각 표현은 작품의 분위기와 장면을 상상하는 데 도움이 되었다."),
    ("q4", "Q4. 설명을 들으며 촉감이나 온도 같은 비시각 감각을 자연스럽게 떠올릴 수 있었다."),
    ("q5", "Q5. 설명을 들으며 작품 속 장면과 분위기에 몰입하는 느낌이 들었다."),
    ("q6", "Q6. 설명의 길이와 정보량은 적절했다."),
    ("q7", "Q7. 문장의 흐름과 속도가 듣기에 편안했다."),
    ("q8", "Q8. 이 설명 방식은 작품 감상에 도움이 되었다."),
    ("q9", "Q9. 실제 미술관이나 전시 서비스에서 이 설명 방식을 사용하고 싶다."),
]

# Likert 9문항을 4개 의미 그룹으로 묶어 화면 표시.
# (LIKERT_QUESTIONS 순서/키 변경 금지 — 응답 CSV 컬럼 매핑이 깨짐)
# 13문항 원안에서 Q6(감정 전달), Q8(표현 복잡도), Q9(음성 자연스러움),
# Q13(장면 선명함)을 중복/저우선순위로 제거하고 9문항으로 재구성한 것.
LIKERT_GROUPS = [
    {
        "title": "1. 머릿속으로 그려보기",
        "subtitle": "구도와 배치",
        "desc": "귀로 듣고 눈으로 보듯, 작품 속 인물이나 물건이 어디에 어떻게 놓여 있는지 쉽게 상상할 수 있었는지 확인하는 문항입니다.",
        "keys": ["q1", "q2"],
    },
    {
        "title": "2. 분위기와 느낌",
        "subtitle": "감각과 몰입",
        "desc": "단순한 사실 전달을 넘어, 작품의 분위기에 푹 빠져들거나 촉감·온도 같은 생생한 느낌을 받았는지 확인하는 문항입니다.",
        "keys": ["q3", "q4", "q5"],
    },
    {
        "title": "3. 듣기의 편안함",
        "subtitle": "설명 양과 속도",
        "desc": "설명이 너무 길거나 짧진 않은지, 문장의 흐름과 속도가 부드러워 듣기 편했는지 확인하는 문항입니다.",
        "keys": ["q6", "q7"],
    },
    {
        "title": "4. 종합 평가",
        "subtitle": "만족도와 추천",
        "desc": "설명을 다 듣고 난 뒤의 최종 결론입니다. 이 방식이 작품 감상에 도움이 되었고 앞으로 미술관에서 또 쓰고 싶은지 묻는 문항입니다.",
        "keys": ["q8", "q9"],
    },
]

PARTICIPANT_GROUPS = [
    "시각장애인 (전맹 또는 저시력 사용자)",
    "일반 사용자",
    "미술/접근성 전문가",
    "기타",
]

BLOCK_CHOICES = ["자동 배정", "A", "B", "C", "D", "E"]
GLOBAL_CHOICES = ["설명 1", "설명 2", "설명 3", "잘 모르겠다"]

# --------------------------------------------------------------------------- #
# CSS (접근성 친화 큰 글씨 / 고대비 / 카드형)
# --------------------------------------------------------------------------- #

CUSTOM_CSS = """
:root, body.dark {
    --sd-bg: #0f172a;
    --sd-card-bg: #1e293b;
    --sd-card-bg-soft: #273449;
    --sd-text: #f1f5f9;
    --sd-text-muted: #cbd5e1;
    --sd-text-dim: #94a3b8;
    --sd-border: #334155;
    --sd-accent: #60a5fa;
    --sd-accent-strong: #3b82f6;
    --sd-accent-bg: rgba(59, 130, 246, 0.2);
    --sd-focus: #93c5fd;
    --sd-warn-bg: rgba(120, 53, 15, 0.4);
    --sd-warn-text: #fcd34d;
    --sd-alert-bg: rgba(190, 18, 60, 0.22);
    --sd-alert-border: #f43f5e;
    --sd-alert-text: #fecdd3;
    --sd-success: #34d399;
}

/* ========= Gradio 라이트 테마 강제 다크화 ========= */
html, body, .gradio-container,
.gradio-container .main, .gradio-container .wrap, .gradio-container .container {
    background: var(--sd-bg) !important;
    color: var(--sd-text) !important;
    font-size: 18px !important;
    line-height: 1.6 !important;
}
.gradio-container { max-width: 980px !important; margin: 0 auto !important; }
.gradio-container * {
    font-family: "Noto Sans KR", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* 모든 텍스트 요소를 밝은 색으로 강제 */
.gradio-container p,
.gradio-container li,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container span,
.gradio-container label,
.gradio-container .prose,
.gradio-container .prose * {
    color: var(--sd-text) !important;
}

/* Gradio 내부 컴포넌트 컨테이너 다크화 */
.gradio-container .form,
.gradio-container .block,
.gradio-container .padded,
.gradio-container .gr-block,
.gradio-container .gr-form,
.gradio-container .gr-padded,
.gradio-container .gr-group {
    background: transparent !important;
    border-color: var(--sd-border) !important;
}

/* 입력 필드 (Textbox, Radio container) */
.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container input:not([type="radio"]):not([type="checkbox"]) {
    background: var(--sd-card-bg-soft) !important;
    color: var(--sd-text) !important;
    border: 1.5px solid var(--sd-border) !important;
    font-size: 17px !important;
}
.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: var(--sd-text-dim) !important;
    opacity: 1;
}
.gradio-container textarea:focus,
.gradio-container input:not([type="radio"]):not([type="checkbox"]):focus {
    border-color: var(--sd-accent) !important;
}

/* Radio 옵션 라벨 — 박스형 (체크박스는 별도 처리) */
.gradio-container .gr-radio,
.gradio-container [data-testid="radio"],
.gradio-container [role="radiogroup"] {
    background: transparent !important;
}
.gradio-container .gr-radio label,
.gradio-container [data-testid="radio"] label,
.gradio-container [role="radiogroup"] label {
    color: var(--sd-text) !important;
    background: var(--sd-card-bg-soft) !important;
    border: 1.5px solid var(--sd-border) !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    min-height: 48px !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    font-size: 16px !important;
}
.gradio-container .gr-radio label:hover,
.gradio-container [role="radiogroup"] label:hover {
    border-color: var(--sd-accent) !important;
}
.gradio-container .gr-radio label:has(input:checked),
.gradio-container [role="radiogroup"] label:has(input:checked) {
    background: var(--sd-accent-bg) !important;
    border-color: var(--sd-accent) !important;
    color: var(--sd-text) !important;
    font-weight: 700 !important;
}

/* Checkbox — 박스화하지 않고 native 체크 마크가 또렷이 보이도록 */
.gradio-container .gr-checkbox,
.gradio-container [data-testid="checkbox"] {
    background: transparent !important;
}
.gradio-container .gr-checkbox label,
.gradio-container [data-testid="checkbox"] label {
    color: var(--sd-text) !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    min-height: auto !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 12px !important;
    font-size: 17px !important;
    cursor: pointer !important;
}

/* 동의 체크박스만 강조 (sd-consent-check) */
.sd-consent-check {
    background: var(--sd-card-bg-soft) !important;
    border: 2px solid var(--sd-accent) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    margin-top: 8px !important;
}
.sd-consent-check label {
    font-weight: 700 !important;
    font-size: 17px !important;
    color: var(--sd-text) !important;
}
.sd-consent-check:has(input:checked) {
    background: var(--sd-accent-bg) !important;
}

/* Block label (gr.Textbox/Radio 의 위 라벨) */
.gradio-container .gr-block-label,
.gradio-container .block-label,
.gradio-container span.label,
.gradio-container label > span:first-child,
.gradio-container .block-title {
    color: var(--sd-text) !important;
    font-weight: 700 !important;
    font-size: 17px !important;
}

/* 버튼 */
.gradio-container button:not(.sd-big-btn):not([class*="audio"]),
.gradio-container .gr-button {
    background: var(--sd-card-bg-soft) !important;
    color: var(--sd-text) !important;
    border: 1.5px solid var(--sd-border) !important;
}
.gradio-container button.primary,
.gradio-container button.gr-button-primary,
.gradio-container .gr-button-primary {
    background: var(--sd-accent-strong) !important;
    color: #ffffff !important;
    border-color: var(--sd-accent-strong) !important;
}

button.sd-big-btn, .sd-big-btn button {
    font-size: 19px !important;
    font-weight: 700 !important;
    padding: 16px 24px !important;
    min-height: 56px !important;
    border-radius: 10px !important;
    letter-spacing: 0.01em;
    width: 100% !important;
}
button.sd-played-confirmed, .sd-played-confirmed button {
    background: var(--sd-success) !important;
    color: #052e1e !important;
    border-color: var(--sd-success) !important;
}

/* 포커스 인디케이터 */
*:focus { outline: none; }
button:focus-visible,
a:focus-visible,
input:focus-visible,
textarea:focus-visible,
[role="button"]:focus-visible,
label:focus-within {
    outline: 3px solid var(--sd-focus) !important;
    outline-offset: 2px !important;
}

/* Skip link */
.sd-skip-link a {
    position: absolute;
    left: -9999px; top: -9999px;
    background: var(--sd-accent-strong);
    color: #fff !important;
    padding: 10px 16px;
    border-radius: 8px;
    font-weight: 700;
    z-index: 10000;
    text-decoration: none;
}
.sd-skip-link a:focus { left: 16px; top: 16px; }

/* 시각 hidden */
.sd-sr-only {
    position: absolute !important;
    width: 1px !important; height: 1px !important;
    padding: 0 !important; margin: -1px !important;
    overflow: hidden !important; clip: rect(0,0,0,0) !important;
    white-space: nowrap !important; border: 0 !important;
}

/* 헤더 */
#sd-header h1 {
    font-size: 28px !important;
    font-weight: 800 !important;
    color: var(--sd-text) !important;
    margin: 12px 0 4px 0 !important;
    letter-spacing: -0.01em;
}
#sd-header .sd-header-sub {
    color: var(--sd-text-muted) !important;
    font-size: 15px;
    margin-bottom: 8px;
}

/* 카드 */
.sd-card {
    background: var(--sd-card-bg) !important;
    border: 1px solid var(--sd-border) !important;
    border-radius: 14px !important;
    padding: 24px !important;
    margin-bottom: 18px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.25);
}

/* 진행률 */
.sd-progress-wrap { display: flex; flex-direction: column; gap: 10px; margin-bottom: 6px; }
.sd-progress-chip {
    display: inline-block;
    background: var(--sd-accent-bg);
    color: var(--sd-text) !important;
    padding: 8px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 17px;
    border: 1px solid var(--sd-accent);
    width: fit-content;
}
.sd-progress-chip strong { color: var(--sd-accent) !important; font-size: 19px; padding: 0 2px; }
.sd-progress-chip .sd-progress-total { color: var(--sd-text-muted) !important; font-weight: 500; }
progress.sd-progress-bar {
    width: 100%; height: 8px;
    border: 0; border-radius: 999px;
    overflow: hidden;
    background: var(--sd-card-bg-soft);
    appearance: none;
}
progress.sd-progress-bar::-webkit-progress-bar { background: var(--sd-card-bg-soft); border-radius: 999px; }
progress.sd-progress-bar::-webkit-progress-value { background: var(--sd-accent-strong); border-radius: 999px; }
progress.sd-progress-bar::-moz-progress-bar { background: var(--sd-accent-strong); border-radius: 999px; }

/* 작품 타이틀 */
.sd-artwork-title {
    font-size: 28px !important;
    font-weight: 800 !important;
    color: var(--sd-text) !important;
    margin: 14px 0 12px 0 !important;
    letter-spacing: -0.01em;
    line-height: 1.3;
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 10px;
}
.sd-artwork-title .sd-title-part:first-child { color: var(--sd-text) !important; }
.sd-artwork-title .sd-title-part:not(:first-child) {
    color: var(--sd-text-muted) !important;
    font-weight: 600 !important;
    font-size: 22px !important;
}
.sd-artwork-title .sd-title-sep {
    color: var(--sd-border) !important;
    font-weight: 400 !important;
    font-size: 22px !important;
}

.sd-label-chip {
    display: inline-block;
    background: var(--sd-accent-strong);
    color: #ffffff !important;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 16px;
    letter-spacing: 0.02em;
    margin-top: 2px;
}

/* 안내 박스 (매 자극물 공통) */
.sd-instructions {
    background: var(--sd-card-bg-soft);
    border: 1px solid var(--sd-border);
    border-left: 4px solid var(--sd-accent);
    color: var(--sd-text) !important;
    padding: 14px 18px;
    border-radius: 10px;
    margin-top: 16px;
    font-size: 16px;
    line-height: 1.65;
}
.sd-instructions strong { color: var(--sd-accent) !important; }

/* 이미지 — 모든 작품 동일한 박스 크기 */
.sd-artwork-image {
    border: 1px solid var(--sd-border) !important;
    border-radius: 12px !important;
    background: var(--sd-card-bg-soft) !important;
    margin-top: 14px;
    height: 460px !important;
    width: 100% !important;
    overflow: hidden !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 8px !important;
}
.sd-artwork-image > div,
.sd-artwork-image .image-container,
.sd-artwork-image .image-frame {
    width: 100% !important;
    height: 100% !important;
    background: transparent !important;
}
.sd-artwork-image img {
    width: 100% !important;
    height: 100% !important;
    max-height: none !important;
    object-fit: contain !important;
    display: block;
}

/* 오디오 그룹 */
.sd-audio-group {
    background: var(--sd-card-bg-soft) !important;
    border: 1px solid var(--sd-border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-top: 16px;
}
.sd-audio-group .sd-audio-label {
    display: block;
    font-size: 17px;
    font-weight: 700;
    color: var(--sd-text) !important;
    margin-bottom: 8px;
}
.sd-play-count {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
    color: var(--sd-text) !important;
    padding: 8px 14px;
    background: var(--sd-card-bg);
    border-radius: 8px;
    border: 1px solid var(--sd-border);
}
.sd-play-count strong { color: var(--sd-accent) !important; font-size: 22px; padding: 0 6px; }

/* 인라인 경고 / 상태 */
.sd-alert {
    background: var(--sd-alert-bg);
    border: 1px solid var(--sd-alert-border);
    color: var(--sd-alert-text) !important;
    padding: 12px 16px;
    border-radius: 10px;
    font-weight: 700;
    margin-bottom: 12px;
}
.sd-alert:empty { display: none; }
.sd-warn {
    color: var(--sd-warn-text) !important;
    background: var(--sd-warn-bg);
    padding: 10px 14px;
    border-radius: 8px;
    font-weight: 600;
    margin-top: 8px;
}
.sd-warn:empty { display: none; padding: 0; }

/* Likert 카드 헤더 */
.sd-likert-heading {
    font-size: 22px !important;
    font-weight: 800 !important;
    color: var(--sd-text) !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.01em;
}
.sd-likert-sub {
    color: var(--sd-text-muted) !important;
    font-size: 15px !important;
    margin: 0 0 18px 0 !important;
}

/* 4개 그룹의 sub-card */
.sd-likert-group {
    background: var(--sd-card-bg-soft) !important;
    border: 1px solid var(--sd-border) !important;
    border-radius: 12px !important;
    padding: 18px !important;
    margin-bottom: 16px !important;
}
.sd-likert-group:last-child { margin-bottom: 0 !important; }

.sd-group-header { margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1px dashed var(--sd-border); }
.sd-group-title {
    margin: 0 0 6px 0 !important;
    font-size: 19px !important;
    font-weight: 800 !important;
    color: var(--sd-accent) !important;
    display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px;
    letter-spacing: -0.005em;
}
.sd-group-title-main { color: var(--sd-accent) !important; }
.sd-group-title-sub {
    color: var(--sd-text-muted) !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}
.sd-group-desc {
    color: var(--sd-text-muted) !important;
    font-size: 14.5px !important;
    line-height: 1.6 !important;
    margin: 0 !important;
}

/* Likert 박스형 그리드 — 문항 라벨 */
.sd-likert label > span:first-child {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: var(--sd-text) !important;
    margin-bottom: 12px !important;
    display: block;
    line-height: 1.5 !important;
    white-space: normal !important;
}
.sd-likert-group .gr-form > div + div { margin-top: 18px !important; }

/* Likert 그리드: radiogroup 컨테이너가 5칸을 가득 채우도록 강제 */
.sd-likert .gr-radio,
.sd-likert [role="radiogroup"],
.sd-likert .wrap,
.sd-likert .form .form,
.sd-likert div[data-testid="radio"] > div {
    display: grid !important;
    grid-template-columns: repeat(5, 1fr) !important;
    gap: 10px !important;
    width: 100% !important;
    max-width: 100% !important;
}

/* 동의 화면의 라디오 그룹 (사용자 그룹, 평가 블록) — 옵션 폭 균등 */
.sd-grid-radio .gr-radio,
.sd-grid-radio [role="radiogroup"],
.sd-grid-radio .wrap,
.sd-grid-radio div[data-testid="radio"] > div {
    display: grid !important;
    gap: 10px !important;
    width: 100% !important;
}
.sd-grid-radio .gr-radio label,
.sd-grid-radio [role="radiogroup"] label {
    width: 100% !important;
    box-sizing: border-box !important;
    margin: 0 !important;
    justify-content: flex-start !important;
}
.sd-grid-radio-2 .gr-radio,
.sd-grid-radio-2 [role="radiogroup"],
.sd-grid-radio-2 .wrap,
.sd-grid-radio-2 div[data-testid="radio"] > div {
    grid-template-columns: repeat(2, 1fr) !important;
}
.sd-grid-radio-6 .gr-radio,
.sd-grid-radio-6 [role="radiogroup"],
.sd-grid-radio-6 .wrap,
.sd-grid-radio-6 div[data-testid="radio"] > div {
    grid-template-columns: repeat(6, 1fr) !important;
}
.sd-grid-radio .block,
.sd-grid-radio .form,
.sd-grid-radio .gr-form,
.sd-grid-radio [data-testid="radio"] {
    width: 100% !important;
}
.sd-likert .gr-radio label,
.sd-likert [role="radiogroup"] label {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    box-sizing: border-box !important;
}
/* 라디오 그룹의 부모 wrapper 들도 width 100% */
.sd-likert .block,
.sd-likert .form,
.sd-likert .gr-form,
.sd-likert .gr-block,
.sd-likert [data-testid="radio"] {
    width: 100% !important;
}
/* Likert 박스형 그리드 — 옵션 자체 외형 */
.sd-likert .gr-radio label,
.sd-likert [role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    padding: 14px 8px !important;
    min-height: 60px !important;
    border: 2px solid var(--sd-border) !important;
    border-radius: 10px !important;
    background: var(--sd-card-bg-soft) !important;
    color: var(--sd-text) !important;
    font-size: 15px !important;
    font-weight: 500 !important;
}

/* 동의 화면 안내 텍스트 강조 */
.sd-consent-text { color: var(--sd-text) !important; font-size: 17px; line-height: 1.7; }
.sd-consent-text h2 { font-size: 22px !important; color: var(--sd-text) !important; margin-top: 0 !important; }
.sd-consent-text strong { color: var(--sd-accent) !important; }
.sd-consent-text ul { margin: 12px 0; padding-left: 24px; }
.sd-consent-text li { margin: 6px 0; color: var(--sd-text) !important; }

/* radio/checkbox 인풋 */
input[type="radio"], input[type="checkbox"] {
    accent-color: var(--sd-accent-strong) !important;
    cursor: pointer;
    flex-shrink: 0;
}
input[type="radio"] {
    width: 22px !important;
    height: 22px !important;
}
input[type="checkbox"] {
    width: 26px !important;
    height: 26px !important;
    /* native 체크 마크가 어두운 배경 위에서 잘 보이도록 */
    background-color: #ffffff;
    border: 2px solid var(--sd-border);
    border-radius: 4px;
    appearance: auto;
    -webkit-appearance: checkbox;
}

/* 모바일 */
@media (max-width: 768px) {
    .gradio-container { padding: 0 8px !important; }
    body, .gradio-container { font-size: 17px !important; }
    .sd-card { padding: 16px !important; }
    #sd-header h1 { font-size: 24px !important; }
    .sd-artwork-title { font-size: 22px !important; }
    .sd-artwork-title .sd-title-part:not(:first-child),
    .sd-artwork-title .sd-title-sep { font-size: 17px !important; }
    .sd-instructions { font-size: 15px; padding: 12px 14px; }
    .sd-artwork-image { height: 320px !important; }
    .sd-likert .gr-radio,
    .sd-likert [role="radiogroup"],
    .sd-likert .wrap,
    .sd-likert .form .form,
    .sd-likert div[data-testid="radio"] > div {
        grid-template-columns: repeat(2, 1fr) !important;
    }
    .sd-likert .gr-radio label { font-size: 14px !important; min-height: 52px !important; padding: 12px 6px !important; }
    .sd-likert-heading { font-size: 19px !important; }
    .sd-group-title { font-size: 17px !important; }
    .sd-group-title-sub { font-size: 14px !important; }
    .sd-group-desc { font-size: 13.5px !important; }
    .sd-likert label > span:first-child { font-size: 16px !important; }
    .sd-likert-group { padding: 14px !important; }
    button.sd-big-btn, .sd-big-btn button { font-size: 17px !important; min-height: 52px !important; }

    /* 동의 화면 라디오 — 평가 블록 6열은 좁아지므로 3열로 wrap */
    .sd-grid-radio-6 .gr-radio,
    .sd-grid-radio-6 [role="radiogroup"],
    .sd-grid-radio-6 .wrap,
    .sd-grid-radio-6 div[data-testid="radio"] > div {
        grid-template-columns: repeat(3, 1fr) !important;
    }
}
@media (max-width: 480px) {
    .sd-likert .gr-radio,
    .sd-likert [role="radiogroup"],
    .sd-likert .wrap,
    .sd-likert .form .form,
    .sd-likert div[data-testid="radio"] > div {
        grid-template-columns: 1fr !important;
    }
    .sd-grid-radio-2 .gr-radio,
    .sd-grid-radio-2 [role="radiogroup"],
    .sd-grid-radio-2 .wrap,
    .sd-grid-radio-2 div[data-testid="radio"] > div {
        grid-template-columns: 1fr !important;
    }
    .sd-grid-radio-6 .gr-radio,
    .sd-grid-radio-6 [role="radiogroup"],
    .sd-grid-radio-6 .wrap,
    .sd-grid-radio-6 div[data-testid="radio"] > div {
        grid-template-columns: repeat(2, 1fr) !important;
    }
}
"""

# --------------------------------------------------------------------------- #
# UI 헬퍼
# --------------------------------------------------------------------------- #


def auto_assign_block(participant_id: str) -> str:
    """participant_id로부터 결정적 block 배정 (해시 기반)."""
    blocks = ["A", "B", "C", "D", "E"]
    key = (participant_id or str(uuid.uuid4())).strip()
    h = sum(ord(c) for c in key)
    return blocks[h % len(blocks)]


def render_stimulus_header(stim: dict, idx: int, total: int) -> tuple[str, str, str, str]:
    """평가 화면 상단 헤더 HTML 4종 반환.

    - progress_html: chip + progress bar
    - title_html: 작품명 / 작가명 / (연도가 있으면) 제작연도 통합 타이틀
    - label_html: <span class="sd-label-chip">설명 1/2/3</span>
    - debug_info: DEBUG_MODE 일 때만 채워짐
    """
    artwork_total = max(total // 3, 1)
    artwork_no = idx // 3 + 1
    desc_no = idx % 3 + 1

    title = stim.get("title", "") or "(작품명 정보 없음)"
    artist = stim.get("artist", "") or ""
    year = stim.get("artwork_year", "")
    year_clean = year if year and year.lower() != "nan" else ""

    title_parts = [title]
    if artist:
        title_parts.append(artist)
    if year_clean:
        title_parts.append(f"{year_clean}")
    combined_title = '<span class="sd-title-sep">/</span>'.join(
        f'<span class="sd-title-part">{p}</span>' for p in title_parts
    )

    blind = stim.get("blind_label", "") or "설명"

    progress_html = (
        f'<div class="sd-progress-wrap">'
        f'  <span class="sd-progress-chip" role="status" aria-live="polite">'
        f'    작품 <strong>{artwork_no}</strong> / {artwork_total}'
        f'    &nbsp;·&nbsp; 설명 <strong>{desc_no}</strong> / 3'
        f'    &nbsp;<span class="sd-progress-total">(총 {idx + 1} / {total})</span>'
        f'  </span>'
        f'  <progress class="sd-progress-bar" max="{total}" value="{idx + 1}" '
        f'aria-label="전체 진행률 {idx + 1} / {total}"></progress>'
        f'</div>'
    )
    title_html = f'<h3 class="sd-artwork-title">{combined_title}</h3>'
    label_html = f'<span class="sd-label-chip">{blind}</span>'

    debug_info = ""
    if DEBUG_MODE:
        debug_info = (
            f"[DEBUG] stimulus_id={stim.get('stimulus_id')} | "
            f"condition={stim.get('condition')} | "
            f"audio={stim.get('audio_file')}\n\n"
            f"script_text:\n{stim.get('script_text', '')}"
        )
    return progress_html, title_html, label_html, debug_info


def play_count_html(count: int) -> str:
    return f'<div class="sd-play-count" aria-live="polite">음성 재생 횟수: <strong>{count}</strong> 회</div>'


def build_audio_state(stim: dict) -> tuple[Optional[str], str]:
    audio_path = resolve_audio_path(stim.get("audio_file", ""))
    if audio_path:
        return audio_path, ""
    return None, '<div class="sd-warn">⚠ 오디오 파일을 찾을 수 없습니다. 진행은 가능하지만 음성 없이 평가합니다.</div>'


def build_image_state(stim: dict) -> tuple[Optional[str], str]:
    image_path = resolve_image_path(stim.get("artwork_id", ""))
    title = stim.get("title", "") or "작품"
    artist = stim.get("artist", "")
    alt = f"작품 이미지: {artist} - {title}" if artist else f"작품 이미지: {title}"
    if image_path:
        return image_path, alt
    return None, alt


# --------------------------------------------------------------------------- #
# 콜백
# --------------------------------------------------------------------------- #


PLAYED_BTN_INITIAL = "🔊 음성을 들었습니다"


def _played_btn_update(count: int):
    if count <= 0:
        return gr.update(
            value=PLAYED_BTN_INITIAL,
            variant="secondary",
            elem_classes=["sd-big-btn"],
        )
    return gr.update(
        value=f"✓ 청취 완료 (재생 횟수: {count})",
        variant="primary",
        elem_classes=["sd-big-btn", "sd-played-confirmed"],
    )


def start_evaluation(
    participant_id,
    participant_group,
    block_choice,
    consent,
    state,
):
    """동의/설정 화면 -> 평가 화면."""
    pid = (participant_id or "").strip()
    if not pid:
        gr.Warning("참가자 ID를 입력해주세요.")
        return _no_advance(state, alert="참가자 ID를 입력해주세요.")
    if not participant_group:
        gr.Warning("사용자 그룹을 선택해주세요.")
        return _no_advance(state, alert="사용자 그룹을 선택해주세요.")
    if not consent:
        gr.Warning("연구 참여 동의 체크가 필요합니다.")
        return _no_advance(state, alert="연구 참여 동의 체크가 필요합니다.")

    block = block_choice if block_choice and block_choice != "자동 배정" else auto_assign_block(pid)
    stimuli = get_block_stimuli(block)
    if not stimuli:
        gr.Warning(f"Block {block}에 자극물이 없습니다. CSV를 확인해주세요.")
        return _no_advance(state, alert=f"Block {block}에 자극물이 없습니다.")

    # 블록의 6작품 중 participant_id 기반 고정 무작위 3작품만 평가 (작품당 3조건 = 9 자극물)
    stimuli = select_artworks_for_participant(stimuli, pid)

    s = SessionState(
        participant_id=pid,
        participant_group=participant_group,
        block_id=block,
        stimuli=stimuli,
        current_index=0,
        audio_play_count=0,
        response_start_time=time.time(),
    )

    stim = s.current()
    progress, title, blind, debug_info = render_stimulus_header(stim, 0, s.total())
    audio_path, audio_warn = build_audio_state(stim)
    image_path, image_alt = build_image_state(stim)

    return (
        s.__dict__,                                  # state
        gr.update(visible=False),                    # consent panel
        gr.update(visible=True),                     # eval panel
        gr.update(visible=False),                    # global panel
        gr.update(visible=False),                    # done panel
        "",                                          # alert (clear)
        progress, title, blind,
        gr.update(value=image_path, label=image_alt),  # image
        audio_path, audio_warn,
        play_count_html(0),                          # play count display
        _played_btn_update(0),                       # played button reset
        debug_info,
        *[gr.update(value=None) for _ in LIKERT_QUESTIONS],
        "", "", "",                                  # free text reset
    )


def _no_advance(state, alert: str = ""):
    """검증 실패 시 화면 유지. alert 가 있으면 인라인 경고만 갱신."""
    alert_html = f'<div class="sd-alert" role="alert" aria-live="assertive">{alert}</div>' if alert else ""
    return (
        state,
        gr.update(),  # consent
        gr.update(),  # eval
        gr.update(),  # global
        gr.update(),  # done
        alert_html,   # alert region
        gr.update(),  # progress
        gr.update(),  # title
        gr.update(),  # blind
        gr.update(),  # image
        gr.update(),  # audio
        gr.update(),  # audio warn
        gr.update(),  # play count
        gr.update(),  # played button
        gr.update(),  # debug
        *[gr.update() for _ in LIKERT_QUESTIONS],
        gr.update(), gr.update(), gr.update(),
    )


def mark_audio_played(state):
    s = state_from_dict(state)
    s.audio_play_count += 1
    return s.__dict__, play_count_html(s.audio_play_count), _played_btn_update(s.audio_play_count)


def submit_response(
    state,
    q1, q2, q3, q4, q5, q6, q7, q8, q9,
    good_expr, awkward_expr, improvement,
):
    """현재 자극물 응답 제출 → 저장 후 다음 자극물 / 글로벌 단계로."""
    s = state_from_dict(state)
    likert_vals = [q1, q2, q3, q4, q5, q6, q7, q8, q9]

    if any(v is None for v in likert_vals):
        msg = "9개의 평가 문항(Q1~Q9)에 모두 응답해주세요."
        gr.Warning(msg)
        return _no_advance(state, alert=msg)

    if s.audio_play_count == 0:
        msg = "음성을 먼저 재생하거나 들은 뒤 '음성을 들었습니다' 버튼을 눌러 주세요."
        gr.Warning(msg)
        return _no_advance(state, alert=msg)

    stim = s.current()
    if stim is None:
        msg = "현재 자극물 정보를 찾을 수 없습니다."
        gr.Warning(msg)
        return _no_advance(state, alert=msg)

    end_time = time.time()
    response_row = {
        "participant_id": s.participant_id,
        "participant_group": s.participant_group,
        "block_id": s.block_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stimulus_index": s.current_index + 1,
        "stimulus_id": stim.get("stimulus_id", ""),
        "artwork_id": stim.get("artwork_id", ""),
        "artist": stim.get("artist", ""),
        "title": stim.get("title", ""),
        "condition": stim.get("condition", ""),                            # 분석용
        "blind_label_for_participant": stim.get("blind_label", ""),
        "audio_file": stim.get("audio_file", ""),
        "q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5,
        "q6": q6, "q7": q7, "q8": q8, "q9": q9,
        "good_expression": (good_expr or "").strip(),
        "awkward_expression": (awkward_expr or "").strip(),
        "improvement_comment": (improvement or "").strip(),
        "audio_play_count": s.audio_play_count,
        "response_start_time": datetime.fromtimestamp(s.response_start_time).isoformat(timespec="seconds"),
        "response_end_time": datetime.fromtimestamp(end_time).isoformat(timespec="seconds"),
        "response_time_sec": round(end_time - s.response_start_time, 2),
    }

    try:
        append_csv(RESPONSES_CSV, RESPONSE_FIELDS, response_row)
    except Exception as e:
        msg = f"응답 저장 중 오류가 발생했습니다: {e}"
        gr.Warning(msg)
        return _no_advance(state, alert=msg)

    # HF Dataset repo 백업 (백그라운드, 실패해도 평가 진행)
    hub_push_response(response_row, kind="stimulus")

    # 다음 자극물로 이동
    s.current_index += 1
    s.audio_play_count = 0
    s.response_start_time = time.time()

    if s.current_index >= s.total():
        # 모든 자극물 완료 → 글로벌 단계로
        return (
            s.__dict__,
            gr.update(visible=False),  # consent
            gr.update(visible=False),  # eval
            gr.update(visible=True),   # global
            gr.update(visible=False),  # done
            "",                        # alert clear
            gr.update(),               # progress
            gr.update(),               # title
            gr.update(),               # blind
            gr.update(value=None),     # image
            gr.update(value=None),     # audio
            gr.update(value=""),       # audio warn
            play_count_html(0),        # play count
            _played_btn_update(0),     # played button reset
            gr.update(value=""),       # debug
            *[gr.update(value=None) for _ in LIKERT_QUESTIONS],
            gr.update(value=""), gr.update(value=""), gr.update(value=""),
        )

    # 다음 자극물 표시
    next_stim = s.current()
    progress, title, blind, debug_info = render_stimulus_header(
        next_stim, s.current_index, s.total()
    )
    audio_path, audio_warn = build_audio_state(next_stim)
    image_path, image_alt = build_image_state(next_stim)

    return (
        s.__dict__,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "",                            # alert clear
        progress, title, blind,
        gr.update(value=image_path, label=image_alt),
        audio_path, audio_warn,
        play_count_html(0),
        _played_btn_update(0),
        debug_info,
        *[gr.update(value=None) for _ in LIKERT_QUESTIONS],
        "", "", "",
    )


def submit_global(state, g1, g2, g3, g4):
    s = state_from_dict(state)
    global_row = {
        "participant_id": s.participant_id,
        "participant_group": s.participant_group,
        "block_id": s.block_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "g1_best_imagery": g1 or "",
        "g2_best_immersion": g2 or "",
        "g3_best_service_use": g3 or "",
        "g4_overall_comment": (g4 or "").strip(),
    }
    try:
        append_csv(GLOBAL_CSV, GLOBAL_FIELDS, global_row)
    except Exception as e:
        gr.Warning(f"전체 응답 저장 중 오류가 발생했습니다: {e}")
        return (
            state,
            gr.update(), gr.update(), gr.update(), gr.update(),
        )

    # HF Dataset repo 백업
    hub_push_response(global_row, kind="global")

    return (
        s.__dict__,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
    )


# --------------------------------------------------------------------------- #
# 디버그 요약
# --------------------------------------------------------------------------- #


def debug_summary() -> str:
    if not DEBUG_MODE:
        return ""
    lines = []
    if LOAD_ERROR:
        lines.append(f"[DEBUG] 데이터 로딩 오류: {LOAD_ERROR}")
    else:
        lines.append(f"[DEBUG] 전체 자극물: {len(STIMULI_DF)}")
        if not STIMULI_DF.empty:
            block_col = resolve_column(STIMULI_DF, "block_id")
            counts = STIMULI_DF[block_col].value_counts().to_dict()
            lines.append(f"[DEBUG] block 별 자극물 수: {counts}")
            # 오디오 존재 확인
            audio_col = resolve_column(STIMULI_DF, "audio_file")
            if audio_col is not None:
                missing = []
                for f in STIMULI_DF[audio_col].dropna().unique():
                    if resolve_audio_path(str(f)) is None:
                        missing.append(str(f))
                if missing:
                    lines.append(f"[DEBUG] 누락된 오디오 파일 {len(missing)}개: {missing[:10]}{'...' if len(missing) > 10 else ''}")
                else:
                    lines.append("[DEBUG] 모든 오디오 파일 존재 확인됨")
            # 이미지 매핑 / 존재 확인
            lines.append(f"[DEBUG] image_mapping.csv 항목 수: {len(IMAGE_MAPPING)}")
            artwork_col = resolve_column(STIMULI_DF, "artwork_id")
            if artwork_col is not None:
                missing_imgs = []
                for aid in STIMULI_DF[artwork_col].dropna().unique():
                    if resolve_image_path(str(aid)) is None:
                        missing_imgs.append(str(aid))
                if missing_imgs:
                    lines.append(
                        f"[DEBUG] 누락된 이미지 {len(missing_imgs)}개 "
                        f"(매핑 또는 images/<artwork_id>.jpg|png|webp 확인 필요): "
                        f"{missing_imgs[:10]}{'...' if len(missing_imgs) > 10 else ''}"
                    )
                else:
                    lines.append("[DEBUG] 모든 작품 이미지 존재 확인됨")
    lines.append(f"[DEBUG] 응답 저장 경로 (로컬): {RESPONSES_CSV}")
    lines.append(f"[DEBUG] 전체 평가 저장 경로 (로컬): {GLOBAL_CSV}")
    if HF_TOKEN and HF_DATASET_REPO:
        lines.append(f"[DEBUG] HF Dataset 백업: {HF_DATASET_REPO} (활성)")
    else:
        lines.append("[DEBUG] HF Dataset 백업: 비활성 (HF_TOKEN, HF_DATASET_REPO 미설정)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #

CONSENT_HTML = """
<div class="sd-consent-text">
  <h2>SenseDocent 사용자 평가 연구</h2>
  <p>본 평가는 시각장애인을 포함한 사용자가 AI 기반 명화 오디오 큐레이션을 어떻게 경험하는지
  확인하기 위한 연구입니다.</p>
  <p>참여자는 작품 설명 음성을 듣고, 각 설명에 대한 <strong>이해도, 감각적 상상, 몰입감,
  청취 편의성, 사용 의향</strong> 등을 평가합니다.</p>
  <ul>
    <li>수집된 응답은 <strong>연구 목적으로만</strong> 사용되며, 분석 시 익명화됩니다.</li>
    <li>평가 도중 <strong>언제든 중단</strong>할 수 있습니다.</li>
    <li>음성 청취가 가능한 조용한 환경에서 진행해 주세요.</li>
    <li>한 명의 참여자는 <strong>6개의 그림 작품</strong>에 대해 <strong>3가지 버전의 다른 음성 설명(큐레이션)</strong>을 듣고 평가합니다.</li>
  </ul>
</div>
"""

GLOBAL_INTRO_HTML = """
<div class="sd-consent-text">
  <h2>전체 평가</h2>
  <p>세 가지 설명 방식 전반에 대한 평가를 진행해주세요.</p>
</div>
"""

DONE_HTML = """
<div class="sd-consent-text" style="text-align:center; padding: 24px 0;">
  <h2 style="font-size:26px !important;">평가가 모두 완료되었습니다 🙏</h2>
  <p>소중한 응답을 주셔서 감사합니다. 응답은 안전하게 저장되었습니다.</p>
  <p>본 연구 결과는 시각장애인을 위한 미술 접근성 향상에 활용될 예정입니다.</p>
  <p>문의: <strong>임수혁 (soohyuk@kakao.com)</strong></p>
</div>
"""


SD_THEME = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    body_text_color="#f1f5f9",
    body_text_color_dark="#f1f5f9",
    body_text_color_subdued="#cbd5e1",
    body_text_color_subdued_dark="#cbd5e1",
    background_fill_primary="#1e293b",
    background_fill_primary_dark="#1e293b",
    background_fill_secondary="#273449",
    background_fill_secondary_dark="#273449",
    border_color_primary="#334155",
    border_color_primary_dark="#334155",
    border_color_accent="#3b82f6",
    block_background_fill="#1e293b",
    block_background_fill_dark="#1e293b",
    block_border_color="#334155",
    block_border_color_dark="#334155",
    block_label_text_color="#f1f5f9",
    block_label_text_color_dark="#f1f5f9",
    block_title_text_color="#f1f5f9",
    block_title_text_color_dark="#f1f5f9",
    input_background_fill="#273449",
    input_background_fill_dark="#273449",
    input_border_color="#334155",
    input_border_color_dark="#334155",
    color_accent="#60a5fa",
    color_accent_soft="rgba(59, 130, 246, 0.2)",
    button_primary_background_fill="#3b82f6",
    button_primary_background_fill_dark="#3b82f6",
    button_primary_background_fill_hover="#2563eb",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#273449",
    button_secondary_background_fill_dark="#273449",
    button_secondary_background_fill_hover="#334155",
    button_secondary_text_color="#f1f5f9",
    button_secondary_text_color_dark="#f1f5f9",
    button_secondary_border_color="#334155",
    button_secondary_border_color_dark="#334155",
    checkbox_background_color="#273449",
    checkbox_background_color_dark="#273449",
    checkbox_background_color_selected="#3b82f6",
    checkbox_background_color_selected_dark="#3b82f6",
    checkbox_label_background_fill="#273449",
    checkbox_label_background_fill_dark="#273449",
    checkbox_label_text_color="#f1f5f9",
    checkbox_label_text_color_dark="#f1f5f9",
)


with gr.Blocks(
    css=CUSTOM_CSS,
    title="SenseDocent 사용자 평가",
    analytics_enabled=False,
    theme=SD_THEME,
    js="() => { document.documentElement.classList.add('dark'); document.body.classList.add('dark'); }",
) as demo:
    state = gr.State(value=empty_state())

    # Skip link (키보드 첫 진입 시 노출)
    gr.HTML(
        '<div class="sd-skip-link"><a href="#sd-main">본문으로 바로 가기</a></div>'
    )

    with gr.Column(elem_id="sd-header"):
        gr.HTML(
            '<h1>SenseDocent 사용자 평가</h1>'
            '<p class="sd-header-sub">AI 명화 오디오 큐레이션 사용자 평가 연구</p>'
        )

    # 데이터 로딩 에러 표시
    if LOAD_ERROR:
        gr.Markdown(
            f"<div class='sd-card'><span class='sd-warn'>⚠ 데이터 로딩 오류:</span><br>{LOAD_ERROR}</div>"
        )

    if DEBUG_MODE:
        gr.Markdown(f"<pre class='sd-card'>{debug_summary()}</pre>")

    # -----------------------------------------------------------------------
    # 화면 1: 동의 / 설정
    # -----------------------------------------------------------------------
    with gr.Column(visible=True) as consent_panel:
        with gr.Column(elem_classes=["sd-card"]):
            gr.HTML(CONSENT_HTML)
            participant_id_in = gr.Textbox(
                label="참가자 ID 또는 이름",
                info="연구자가 별도로 부여한 ID가 있다면 그 ID를, 없다면 이름을 입력해 주세요.",
                placeholder="예: P001 또는 홍길동",
                interactive=True,
            )
            participant_group_in = gr.Radio(
                choices=PARTICIPANT_GROUPS,
                label="사용자 그룹",
                value=None,
                elem_classes=["sd-grid-radio", "sd-grid-radio-2"],
            )
            block_in = gr.Radio(
                choices=BLOCK_CHOICES,
                label="평가 블록 (자동 배정 권장)",
                value="자동 배정",
                elem_classes=["sd-grid-radio", "sd-grid-radio-6"],
            )
            consent_in = gr.Checkbox(
                label="위 내용을 확인했으며 연구 참여에 동의합니다.",
                value=False,
                elem_classes=["sd-consent-check"],
            )
            start_btn = gr.Button("평가 시작", variant="primary", elem_classes=["sd-big-btn"])

    # -----------------------------------------------------------------------
    # 화면 2: 평가
    # -----------------------------------------------------------------------
    with gr.Column(visible=False, elem_id="sd-main") as eval_panel:
        gr.HTML('<h2 class="sd-sr-only">자극물 평가</h2>')
        alert_md = gr.HTML("")

        with gr.Column(elem_classes=["sd-card"]):
            progress_md = gr.HTML("")
            title_md = gr.HTML("")
            blind_md = gr.HTML("")

            gr.HTML(
                '<div class="sd-instructions" role="note">'
                '아래 작품 이미지를 잠시 살펴보신 뒤, 같은 작품에 대한 '
                '<strong>세 가지 버전의 음성 설명</strong>을 차례로 들어주세요. '
                '음성을 듣고 난 뒤 아래 <strong>9개 평가 문항(1~5점)</strong>에 응답해 주시면 됩니다. '
                '자유 응답은 선택 입력입니다.'
                '</div>'
            )

            image_display = gr.Image(
                label="작품 이미지",
                interactive=False,
                show_label=False,
                height=480,
                elem_classes=["sd-artwork-image"],
            )

            with gr.Column(elem_classes=["sd-audio-group"]):
                gr.HTML(
                    '<span class="sd-audio-label">음성 설명</span>'
                    '<span class="sd-sr-only">'
                    "재생 버튼을 눌러 들으세요. 들은 뒤 아래 '음성을 들었습니다' 버튼을 눌러 주세요."
                    "</span>"
                )
                audio_player = gr.Audio(
                    label="음성 설명",
                    interactive=False,
                    show_label=False,
                    autoplay=False,
                )
                audio_warn_md = gr.HTML("")

                with gr.Row():
                    played_btn = gr.Button(
                        PLAYED_BTN_INITIAL,
                        variant="secondary",
                        elem_classes=["sd-big-btn"],
                    )
                    play_count_md = gr.HTML(play_count_html(0))

            debug_md = gr.Markdown("", visible=DEBUG_MODE)

        # Likert 13문항 — 4개 그룹으로 묶어 표시 (LIKERT_QUESTIONS 순서 유지)
        likert_inputs_by_key: dict = {}
        question_label_by_key = {k: q for k, q in LIKERT_QUESTIONS}

        with gr.Column(elem_classes=["sd-card", "sd-likert"]):
            gr.HTML(
                '<h3 class="sd-likert-heading">아래 음성 설명에 대한 평가</h3>'
                '<p class="sd-likert-sub">'
                "각 문항을 읽고 1점(매우 아니다) ~ 5점(매우 그렇다) 중 가장 가까운 것을 선택해 주세요."
                "</p>"
            )

            for group in LIKERT_GROUPS:
                with gr.Column(elem_classes=["sd-likert-group"]):
                    gr.HTML(
                        f'<div class="sd-group-header">'
                        f'  <h4 class="sd-group-title">'
                        f'    <span class="sd-group-title-main">{group["title"]}</span>'
                        f'    <span class="sd-group-title-sub">({group["subtitle"]})</span>'
                        f'  </h4>'
                        f'  <p class="sd-group-desc">{group["desc"]}</p>'
                        f'</div>'
                    )
                    for key in group["keys"]:
                        r = gr.Radio(
                            choices=LIKERT_CHOICES,
                            label=question_label_by_key[key],
                            value=None,
                        )
                        likert_inputs_by_key[key] = r

        # 콜백 입력 순서를 LIKERT_QUESTIONS 순서로 유지 (q1, q2, ..., q13)
        likert_inputs = [likert_inputs_by_key[k] for k, _ in LIKERT_QUESTIONS]

        with gr.Column(elem_classes=["sd-card"]):
            gr.HTML(
                '<h3 style="margin:0 0 12px 0; font-size:20px;">'
                '자유 응답 '
                '<span style="color:var(--sd-text-muted); font-weight:500; font-size:16px;">'
                '(선택 입력)'
                '</span></h3>'
            )
            good_expr = gr.Textbox(
                label="가장 이해하기 쉬웠거나 좋았던 표현이 있다면 적어주세요.",
                placeholder="예) '소용돌이' 표현이 인상적이었어요",
                lines=2,
            )
            awkward_expr = gr.Textbox(
                label="이해하기 어렵거나 어색했던 표현이 있다면 적어주세요.",
                placeholder="예) '대각선 구도'가 무슨 뜻인지 잘 떠오르지 않았어요",
                lines=2,
            )
            improvement = gr.Textbox(
                label="추가로 개선되었으면 하는 점이 있다면 적어주세요.",
                placeholder="설명 속도, 비유, 묘사 방식 등에 대한 의견을 자유롭게 적어 주세요.",
                lines=2,
            )

        next_btn = gr.Button("다음 설명 듣기 →", variant="primary", elem_classes=["sd-big-btn"])

    # -----------------------------------------------------------------------
    # 화면 3: 전체 평가
    # -----------------------------------------------------------------------
    with gr.Column(visible=False) as global_panel:
        with gr.Column(elem_classes=["sd-card"]):
            gr.HTML(GLOBAL_INTRO_HTML)
            g1 = gr.Radio(
                choices=GLOBAL_CHOICES,
                label="G1. 세 가지 설명 방식 중 가장 작품이 잘 상상되었던 설명은 무엇이었나요?",
                value=None,
            )
            g2 = gr.Radio(
                choices=GLOBAL_CHOICES,
                label="G2. 세 가지 설명 방식 중 가장 몰입감이 높았던 설명은 무엇이었나요?",
                value=None,
            )
            g3 = gr.Radio(
                choices=GLOBAL_CHOICES,
                label="G3. 세 가지 설명 방식 중 실제 서비스로 가장 사용하고 싶은 설명은 무엇이었나요?",
                value=None,
            )
            g4 = gr.Textbox(
                label="G4. 전체 평가를 하며 느낀 점이나 개선 의견을 자유롭게 적어주세요.",
                lines=4,
            )
            submit_btn = gr.Button("최종 제출", variant="primary", elem_classes=["sd-big-btn"])

    # -----------------------------------------------------------------------
    # 화면 4: 완료
    # -----------------------------------------------------------------------
    with gr.Column(visible=False) as done_panel:
        with gr.Column(elem_classes=["sd-card"]):
            gr.HTML(DONE_HTML)

    # -------- 이벤트 연결 --------
    start_outputs = [
        state,
        consent_panel, eval_panel, global_panel, done_panel,
        alert_md,
        progress_md, title_md, blind_md,
        image_display,
        audio_player, audio_warn_md,
        play_count_md,
        played_btn,
        debug_md,
        *likert_inputs,
        good_expr, awkward_expr, improvement,
    ]

    start_btn.click(
        fn=start_evaluation,
        inputs=[participant_id_in, participant_group_in, block_in, consent_in, state],
        outputs=start_outputs,
    )

    played_btn.click(
        fn=mark_audio_played,
        inputs=[state],
        outputs=[state, play_count_md, played_btn],
    )

    next_btn.click(
        fn=submit_response,
        inputs=[state, *likert_inputs, good_expr, awkward_expr, improvement],
        outputs=start_outputs,
    )

    submit_btn.click(
        fn=submit_global,
        inputs=[state, g1, g2, g3, g4],
        outputs=[state, consent_panel, eval_panel, global_panel, done_panel],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        ssr_mode=False,
        allowed_paths=[str(AUDIO_DIR), str(IMAGES_DIR), str(IMAGES_DIR_ALT)],
    )
