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
import os
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
    "q10",
    "q11",
    "q12",
    "q13",
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
    ("q6", "Q6. 설명을 통해 작품의 감정이나 분위기가 전달되었다."),
    ("q7", "Q7. 설명의 길이와 정보량은 적절했다."),
    ("q8", "Q8. 설명 속 표현은 이해하기 어렵거나 과도하게 복잡하지 않았다."),
    ("q9", "Q9. 음성으로 듣기에 자연스러운 설명이었다."),
    ("q10", "Q10. 문장의 흐름과 속도가 듣기에 편안했다."),
    ("q11", "Q11. 이 설명 방식은 작품 감상에 도움이 되었다."),
    ("q12", "Q12. 실제 미술관이나 전시 서비스에서 이 설명 방식을 사용하고 싶다."),
    ("q13", "Q13. 설명을 들은 뒤 작품 장면이 머릿속에 비교적 선명하게 떠올랐다."),
]

PARTICIPANT_GROUPS = [
    "시각장애인/저시력 사용자",
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
:root {
    --sd-bg: #0f172a;
    --sd-card-bg: #1e293b;
    --sd-card-bg-soft: #273449;
    --sd-text: #f1f5f9;
    --sd-text-muted: #94a3b8;
    --sd-border: #334155;
    --sd-accent: #60a5fa;
    --sd-accent-strong: #3b82f6;
    --sd-accent-bg: rgba(59, 130, 246, 0.18);
    --sd-focus: #93c5fd;
    --sd-warn-bg: rgba(120, 53, 15, 0.35);
    --sd-warn-text: #fcd34d;
    --sd-alert-bg: rgba(190, 18, 60, 0.18);
    --sd-alert-border: #f43f5e;
    --sd-alert-text: #fda4af;
    --sd-success: #34d399;
}

html, body, .gradio-container {
    background: var(--sd-bg) !important;
    color: var(--sd-text) !important;
    font-size: 18px !important;
    line-height: 1.6 !important;
}

.gradio-container { max-width: 980px !important; margin: 0 auto !important; }

.gradio-container * {
    font-family: "Noto Sans KR", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* ----- 포커스 인디케이터 (키보드 접근성) ----- */
*:focus { outline: none; }
button:focus-visible,
a:focus-visible,
input:focus-visible,
textarea:focus-visible,
[role="button"]:focus-visible,
.gr-radio label:focus-within,
.sd-likert label:focus-within {
    outline: 3px solid var(--sd-focus) !important;
    outline-offset: 2px !important;
    border-radius: 6px;
}

/* ----- Skip link (키보드 첫 진입) ----- */
.sd-skip-link a {
    position: absolute;
    left: -9999px;
    top: -9999px;
    background: var(--sd-accent-strong);
    color: #fff !important;
    padding: 10px 16px;
    border-radius: 8px;
    font-weight: 700;
    z-index: 10000;
    text-decoration: none;
}
.sd-skip-link a:focus { left: 16px; top: 16px; }

/* ----- 시각적 hidden (스크린리더에는 노출) ----- */
.sd-sr-only {
    position: absolute !important;
    width: 1px !important; height: 1px !important;
    padding: 0 !important; margin: -1px !important;
    overflow: hidden !important; clip: rect(0,0,0,0) !important;
    white-space: nowrap !important; border: 0 !important;
}

/* ----- 헤더 ----- */
#sd-header h1 {
    font-size: 28px !important;
    font-weight: 800 !important;
    color: var(--sd-text) !important;
    margin: 12px 0 4px 0 !important;
    letter-spacing: -0.01em;
}
#sd-header .sd-header-sub {
    color: var(--sd-text-muted);
    font-size: 15px;
    margin-bottom: 8px;
}

/* ----- 카드 ----- */
.sd-card {
    background: var(--sd-card-bg) !important;
    border: 1px solid var(--sd-border) !important;
    border-radius: 14px !important;
    padding: 24px !important;
    margin-bottom: 18px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.25);
}

/* ----- 진행률 chip + bar ----- */
.sd-progress-wrap { display: flex; flex-direction: column; gap: 10px; margin-bottom: 6px; }
.sd-progress-chip {
    display: inline-block;
    background: var(--sd-accent-bg);
    color: var(--sd-accent);
    padding: 8px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 17px;
    border: 1px solid var(--sd-accent);
    width: fit-content;
}
.sd-progress-chip strong { color: var(--sd-text); }
progress.sd-progress-bar {
    width: 100%;
    height: 8px;
    border: 0;
    border-radius: 999px;
    overflow: hidden;
    background: var(--sd-card-bg-soft);
}
progress.sd-progress-bar::-webkit-progress-bar { background: var(--sd-card-bg-soft); border-radius: 999px; }
progress.sd-progress-bar::-webkit-progress-value { background: var(--sd-accent-strong); border-radius: 999px; }
progress.sd-progress-bar::-moz-progress-bar { background: var(--sd-accent-strong); border-radius: 999px; }

/* ----- 작품 타이틀 / 작가 / 설명 라벨 ----- */
.sd-artwork-title {
    font-size: 30px !important;
    font-weight: 800 !important;
    color: var(--sd-text) !important;
    margin: 14px 0 4px 0 !important;
    letter-spacing: -0.01em;
    line-height: 1.25;
}
.sd-artist {
    font-size: 19px !important;
    color: var(--sd-text-muted) !important;
    margin: 0 0 14px 0 !important;
}
.sd-label-chip {
    display: inline-block;
    background: var(--sd-accent-strong);
    color: #fff !important;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 16px;
    letter-spacing: 0.02em;
}

/* ----- 이미지 ----- */
.sd-artwork-image {
    border: 1px solid var(--sd-border) !important;
    border-radius: 12px !important;
    background: var(--sd-card-bg-soft) !important;
    margin-top: 14px;
}
.sd-artwork-image img {
    max-width: 100% !important;
    max-height: 480px !important;
    height: auto !important;
    object-fit: contain !important;
    display: block;
    margin: 0 auto;
}

/* ----- 오디오 영역 그룹 ----- */
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
    color: var(--sd-text);
    margin-bottom: 8px;
}
.sd-play-count {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
    color: var(--sd-text);
    padding: 8px 14px;
}
.sd-play-count strong { color: var(--sd-accent); font-size: 20px; padding: 0 4px; }

/* ----- 버튼 ----- */
button.sd-big-btn, .sd-big-btn button {
    font-size: 19px !important;
    font-weight: 700 !important;
    padding: 16px 24px !important;
    min-height: 56px !important;
    border-radius: 10px !important;
    letter-spacing: 0.01em;
}
button.sd-played-confirmed, .sd-played-confirmed button {
    background: var(--sd-success) !important;
    color: #052e1e !important;
    border-color: var(--sd-success) !important;
}

/* ----- 인라인 경고 / 상태 ----- */
.sd-alert {
    background: var(--sd-alert-bg);
    border: 1px solid var(--sd-alert-border);
    color: var(--sd-alert-text);
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

/* ----- Likert 박스형 그리드 ----- */
.sd-likert .gr-form { gap: 22px; }
.sd-likert label > span:first-child {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: var(--sd-text) !important;
    margin-bottom: 10px !important;
    line-height: 1.45;
}
.sd-likert .gr-radio,
.sd-likert .gr-form > div > div[role="radiogroup"] {
    display: grid !important;
    grid-template-columns: repeat(5, 1fr) !important;
    gap: 10px !important;
}
.sd-likert label[data-testid="radio-option"],
.sd-likert .gr-radio label {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center;
    padding: 14px 8px !important;
    min-height: 56px !important;
    border: 2px solid var(--sd-border) !important;
    border-radius: 10px !important;
    background: var(--sd-card-bg-soft) !important;
    color: var(--sd-text) !important;
    font-size: 15px !important;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
}
.sd-likert .gr-radio label:hover { border-color: var(--sd-accent); }
.sd-likert .gr-radio label:has(input:checked),
.sd-likert .gr-radio label.selected {
    background: var(--sd-accent-bg) !important;
    border-color: var(--sd-accent) !important;
    color: var(--sd-text) !important;
    font-weight: 700 !important;
}

/* ----- 동의/그룹 라디오, 텍스트박스, 체크박스 ----- */
.gr-textbox textarea, .gr-textbox input, textarea, input[type="text"] {
    background: var(--sd-card-bg-soft) !important;
    color: var(--sd-text) !important;
    border: 1.5px solid var(--sd-border) !important;
    font-size: 17px !important;
}
.gr-textbox textarea:focus, input[type="text"]:focus {
    border-color: var(--sd-accent) !important;
}
label > span, .gr-checkbox label, .gr-radio label > span {
    color: var(--sd-text) !important;
    font-size: 17px !important;
}

input[type="radio"], input[type="checkbox"] {
    width: 22px !important;
    height: 22px !important;
    accent-color: var(--sd-accent-strong);
}

/* ----- Gradio 컴포넌트 다크 대응 ----- */
.gradio-container .form, .gradio-container .block { background: transparent !important; border: 0 !important; }
.gradio-container .gr-padded { padding: 0 !important; }

/* ----- 모바일 ----- */
@media (max-width: 768px) {
    .gradio-container { padding: 0 8px !important; }
    body, .gradio-container { font-size: 17px !important; }
    .sd-card { padding: 16px !important; }
    #sd-header h1 { font-size: 24px !important; }
    .sd-artwork-title { font-size: 24px !important; }
    .sd-artist { font-size: 17px !important; }
    .sd-likert .gr-radio,
    .sd-likert .gr-form > div > div[role="radiogroup"] {
        grid-template-columns: repeat(2, 1fr) !important;
    }
    .sd-likert .gr-radio label { font-size: 14px !important; min-height: 52px !important; padding: 12px 6px !important; }
    button.sd-big-btn, .sd-big-btn button { font-size: 17px !important; min-height: 52px !important; }
    .sd-artwork-image img { max-height: 360px !important; }
}
@media (max-width: 480px) {
    .sd-likert .gr-radio,
    .sd-likert .gr-form > div > div[role="radiogroup"] {
        grid-template-columns: 1fr !important;
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


def render_stimulus_header(stim: dict, idx: int, total: int) -> tuple[str, str, str, str, str]:
    """평가 화면 상단 헤더 HTML 5종 반환.

    - progress_html: chip + progress bar (aria-live polite 영역 안에 표시됨)
    - title_html: <h3 class="sd-artwork-title">…</h3>
    - artist_html: <p class="sd-artist">…</p>
    - label_html: <span class="sd-label-chip">…</span>
    - debug_info: DEBUG_MODE 일 때만 채워짐
    """
    # 블록 내 자극물은 작품별로 3개씩 정렬되어 있다는 전제 (get_block_stimuli 가 보장).
    artwork_total = max(total // 3, 1)
    artwork_no = idx // 3 + 1
    desc_no = idx % 3 + 1

    title = stim.get("title", "") or "(작품명 정보 없음)"
    artist = stim.get("artist", "") or ""
    year = stim.get("artwork_year", "")
    artist_line = artist + (f" · {year}" if year and year.lower() != "nan" else "")
    blind = stim.get("blind_label", "") or "설명"

    progress_html = (
        f'<div class="sd-progress-wrap">'
        f'  <span class="sd-progress-chip" role="status" aria-live="polite">'
        f'    작품 <strong>{artwork_no}</strong> / {artwork_total}'
        f'    &nbsp;·&nbsp; 설명 <strong>{desc_no}</strong> / 3'
        f'    &nbsp;<span style="opacity:0.75">(총 {idx + 1} / {total})</span>'
        f'  </span>'
        f'  <progress class="sd-progress-bar" max="{total}" value="{idx + 1}" '
        f'aria-label="전체 진행률 {idx + 1} / {total}"></progress>'
        f'</div>'
    )
    title_html = f'<h3 class="sd-artwork-title">{title}</h3>'
    artist_html = f'<p class="sd-artist">{artist_line}</p>' if artist_line else '<p class="sd-artist"></p>'
    label_html = f'<span class="sd-label-chip">{blind}</span>'

    debug_info = ""
    if DEBUG_MODE:
        debug_info = (
            f"[DEBUG] stimulus_id={stim.get('stimulus_id')} | "
            f"condition={stim.get('condition')} | "
            f"audio={stim.get('audio_file')}\n\n"
            f"script_text:\n{stim.get('script_text', '')}"
        )
    return progress_html, title_html, artist_html, label_html, debug_info


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
    progress, title, artist_line, blind, debug_info = render_stimulus_header(stim, 0, s.total())
    audio_path, audio_warn = build_audio_state(stim)
    image_path, image_alt = build_image_state(stim)

    return (
        s.__dict__,                                  # state
        gr.update(visible=False),                    # consent panel
        gr.update(visible=True),                     # eval panel
        gr.update(visible=False),                    # global panel
        gr.update(visible=False),                    # done panel
        "",                                          # alert (clear)
        progress, title, artist_line, blind,
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
        gr.update(),  # artist
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
    q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13,
    good_expr, awkward_expr, improvement,
):
    """현재 자극물 응답 제출 → 저장 후 다음 자극물 / 글로벌 단계로."""
    s = state_from_dict(state)
    likert_vals = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13]

    if any(v is None for v in likert_vals):
        msg = "13개의 평가 문항(Q1~Q13)에 모두 응답해주세요."
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
        "q6": q6, "q7": q7, "q8": q8, "q9": q9, "q10": q10,
        "q11": q11, "q12": q12, "q13": q13,
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
            gr.update(),               # artist
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
    progress, title, artist_line, blind, debug_info = render_stimulus_header(
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
        progress, title, artist_line, blind,
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
    lines.append(f"[DEBUG] 응답 저장 경로: {RESPONSES_CSV}")
    lines.append(f"[DEBUG] 전체 평가 저장 경로: {GLOBAL_CSV}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #

CONSENT_TEXT = """
## SenseDocent 사용자 평가 연구

본 평가는 시각장애인을 포함한 사용자가 AI 기반 명화 오디오 큐레이션을 어떻게 경험하는지
확인하기 위한 연구입니다.

참여자는 작품 설명 음성을 듣고, 각 설명에 대한 **이해도, 감각적 상상, 몰입감,
청취 편의성, 사용 의향** 등을 평가합니다.

- 수집된 응답은 **연구 목적으로만** 사용되며, 분석 시 익명화됩니다.
- 평가 도중 **언제든 중단**할 수 있습니다.
- 음성 청취가 가능한 조용한 환경에서 진행해 주세요.
- 한 명의 참여자는 한 블록(약 18개 음성 자극물)을 평가합니다.
"""

EVAL_INTRO = """
각 음성 설명을 들은 후, 아래 13개 평가 문항(Q1~Q13)에 1~5점으로 응답해 주세요.
자유 응답(F1~F3)은 선택 입력입니다.
"""

GLOBAL_INTRO = """
## 전체 평가

세 가지 설명 방식 전반에 대한 평가를 진행해주세요.
"""

DONE_TEXT = """
## 평가가 모두 완료되었습니다 🙏

소중한 응답을 주셔서 감사합니다. 응답은 안전하게 저장되었습니다.

본 연구 결과는 시각장애인을 위한 미술 접근성 향상에 활용될 예정입니다.

문의: **연구자 이메일 (placeholder@example.com)**
"""


with gr.Blocks(
    css=CUSTOM_CSS,
    title="SenseDocent 사용자 평가",
    analytics_enabled=False,
    theme=gr.themes.Base(),
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
            gr.Markdown(CONSENT_TEXT)
            participant_id_in = gr.Textbox(
                label="참가자 ID (예: P001)",
                placeholder="연구자가 부여한 ID 또는 임의의 식별자",
                interactive=True,
            )
            participant_group_in = gr.Radio(
                choices=PARTICIPANT_GROUPS,
                label="사용자 그룹",
                value=None,
            )
            block_in = gr.Radio(
                choices=BLOCK_CHOICES,
                label="평가 블록 (자동 배정 권장)",
                value="자동 배정",
            )
            consent_in = gr.Checkbox(
                label="위 내용을 확인했으며 연구 참여에 동의합니다.",
                value=False,
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
            artist_md = gr.HTML("")
            blind_md = gr.HTML("")

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

            gr.HTML(
                '<p style="color:var(--sd-text-muted); font-size:15px; margin-top:14px;">'
                "음성 설명을 들은 뒤, 아래 13개 평가 문항에 1~5점으로 응답해 주세요. "
                "자유 응답은 선택 입력입니다."
                "</p>"
            )

        likert_inputs = []
        with gr.Column(elem_classes=["sd-card", "sd-likert"]):
            for key, q in LIKERT_QUESTIONS:
                r = gr.Radio(
                    choices=LIKERT_CHOICES,
                    label=q,
                    value=None,
                )
                likert_inputs.append(r)

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

        next_btn = gr.Button("다음 자극물로 →", variant="primary", elem_classes=["sd-big-btn"])

    # -----------------------------------------------------------------------
    # 화면 3: 전체 평가
    # -----------------------------------------------------------------------
    with gr.Column(visible=False) as global_panel:
        with gr.Column(elem_classes=["sd-card"]):
            gr.Markdown(GLOBAL_INTRO)
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
            gr.Markdown(DONE_TEXT)

    # -------- 이벤트 연결 --------
    start_outputs = [
        state,
        consent_panel, eval_panel, global_panel, done_panel,
        alert_md,
        progress_md, title_md, artist_md, blind_md,
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
