import json
import os
import re
import ast
import uuid
from io import BytesIO
from urllib.parse import quote
from typing import Any
from datetime import datetime, timedelta

import ollama
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scipy import stats
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

app = FastAPI(title="Data Lens API", version="1.0.0")


def _build_origins() -> list[str]:
    origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://sw-cap2.s3-website.ap-northeast-2.amazonaws.com",
        "https://d2pqx6mndt1m20.cloudfront.net",
    ]

    # Optional production origins (set on EC2 environment)
    s3_origin = os.environ.get("FRONTEND_S3_ORIGIN", "").strip()
    cloudfront_origin = os.environ.get("FRONTEND_CLOUDFRONT_ORIGIN", "").strip()
    extra_origins_raw = os.environ.get("FRONTEND_ORIGINS", "").strip()

    if s3_origin:
        origins.append(s3_origin)
    if cloudfront_origin:
        origins.append(cloudfront_origin)
    if extra_origins_raw:
        origins.extend([o.strip() for o in extra_origins_raw.split(",") if o.strip()])

    # Deduplicate while preserving order
    return list(dict.fromkeys(origins))


origins = _build_origins()

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./datalens.db")
IS_SQLITE = DATABASE_URL.startswith("sqlite")
ENGINE = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if IS_SQLITE else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)
Base = declarative_base()


class AnalysisReportModel(Base):
    __tablename__ = "analysis_reports"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    question = Column(Text, nullable=False)
    recommended_method = Column(String(255), nullable=False)
    explanation = Column(Text, nullable=False)
    insights_json = Column(Text, nullable=False)
    chart_data_json = Column(Text, nullable=False)
    table_data_json = Column(Text, nullable=False)
    summary_json = Column(Text, nullable=False)
    evidence_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ReportUpdateRequest(BaseModel):
    file_name: str | None = None


class AssistantAskRequest(BaseModel):
    report_id: int
    question: str


class SessionAnalyzeRequest(BaseModel):
    session_id: str
    question: str = "데이터의 핵심 인사이트를 설명해줘"


class SessionEditRequest(BaseModel):
    session_id: str
    command: str
    question: str | None = None


class SessionResetRequest(BaseModel):
    session_id: str
    question: str | None = None


Base.metadata.create_all(bind=ENGINE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
너는 데이터 분석 어시스턴트다.
반드시 JSON만 반환하고 설명 텍스트를 JSON 바깥에 쓰지 마라.

출력 스키마:
{
  "recommended_method": "분석기법명",
  "explanation": "4~7문장 한국어 설명",
  "insights": ["핵심 인사이트 1", "핵심 인사이트 2", "핵심 인사이트 3"],
  "chart_data": [
    {"category": "범주명", "value": 0}
  ]
}

규칙:
1) 반드시 한국어만 사용
2) 제공된 통계 요약 외 임의 수치 생성 금지
3) p-value가 0.05보다 작으면 유의미하다고 설명
4) 개인정보(이름, 전화번호, 이메일, 주소)는 절대 출력 금지
5) 설명은 보고서 톤으로 자연스럽고 문법적으로 정확하게 작성
""".strip()

EDIT_SYSTEM_PROMPT = """
너는 Pandas 데이터 수정 전문가다.
반드시 파이썬 코드 한 줄만 반환하라.

규칙:
1) 반드시 df 변수에 재할당하는 한 줄 코드만 출력 (예: df = df[df['나이'] > 20])
2) import, 함수 정의, 반복문, 여러 줄 코드 금지
3) 파일 입출력, 네트워크 호출, 시스템 명령 금지
4) 컬럼명은 제공된 컬럼 목록만 사용
5) 사용자가 삭제/제외를 요청하면 해당 조건을 제거하는 필터를 작성
6) 한국어 설명 문장 없이 코드만 출력
""".strip()


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(min_value, int(raw))
    except Exception:
        return default


MAX_GRID_ROWS = _env_int("DATALENS_GRID_MAX_ROWS", 20000)
SESSION_TTL_MINUTES = _env_int("DATALENS_SESSION_TTL_MINUTES", 180)
DATAFRAME_SESSIONS: dict[str, dict[str, Any]] = {}

_BANNED_CODE_PATTERNS = [
    r"__",
    r"\bimport\b",
    r"\bopen\b",
    r"\bexec\b",
    r"\beval\b",
    r"\bos\b",
    r"\bsys\b",
    r"\bsubprocess\b",
    r"\bpathlib\b",
    r"\bpickle\b",
    r"\bread_(csv|excel|sql)\b",
    r"\bto_(csv|excel|sql)\b",
]

_ALLOWED_NAME_IDS = {"df", "pd", "np", "True", "False", "None"}


MASK_KEYWORDS = [
    "name",
    "이름",
    "성명",
    "phone",
    "전화",
    "휴대폰",
    "mobile",
    "email",
    "메일",
    "address",
    "주소",
    "주민",
    "rrn",
    "ssn",
]


def _is_sensitive_col(col_name: str) -> bool:
    lowered = col_name.lower()
    return any(key in lowered for key in MASK_KEYWORDS)


def _mask_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    masked = df.copy()
    for col in masked.columns:
        if _is_sensitive_col(str(col)):
            masked[col] = "***"
    return masked


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    # 헤더/문자열 공백 정리 + 빈 문자열을 결측으로 통일
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    cleaned = cleaned.replace(r"^\s*$", np.nan, regex=True)

    # 완전히 비어 있는 행/열 제거
    cleaned = cleaned.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # 엑셀에서 흔히 생기는 Unnamed 열 제거
    unnamed_cols = [c for c in cleaned.columns if str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        cleaned = cleaned.drop(columns=unnamed_cols)

    # object 열은 좌우 공백 제거 후 빈 문자열을 결측 처리
    obj_cols = cleaned.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        for c in obj_cols:
            cleaned[c] = cleaned[c].map(lambda v: v.strip() if isinstance(v, str) else v)
            cleaned[c] = cleaned[c].replace("", np.nan)

    return cleaned


def _utcnow() -> datetime:
    return datetime.utcnow()


def _prune_sessions() -> None:
    if not DATAFRAME_SESSIONS:
        return
    now = _utcnow()
    expire_before = now - timedelta(minutes=SESSION_TTL_MINUTES)
    stale_ids = [
        sid
        for sid, entry in DATAFRAME_SESSIONS.items()
        if isinstance(entry.get("updated_at"), datetime) and entry["updated_at"] < expire_before
    ]
    for sid in stale_ids:
        DATAFRAME_SESSIONS.pop(sid, None)


def _create_dataframe_session(file_name: str, df: pd.DataFrame) -> str:
    _prune_sessions()
    session_id = str(uuid.uuid4())
    DATAFRAME_SESSIONS[session_id] = {
        "file_name": file_name,
        "original_df": df.copy(deep=True),
        "df": df.copy(deep=True),
        "is_modified": False,
        "updated_at": _utcnow(),
    }
    return session_id


def _get_dataframe_session(session_id: str) -> dict[str, Any]:
    _prune_sessions()
    entry = DATAFRAME_SESSIONS.get(session_id)
    if not entry:
        raise HTTPException(
            status_code=404,
            detail="데이터 세션을 찾을 수 없습니다. 파일을 다시 업로드해 주세요.",
        )
    entry["updated_at"] = _utcnow()
    return entry


def _to_jsonable_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _build_grid_payload(df: pd.DataFrame, prefix: str = "grid") -> dict[str, Any]:
    masked = _mask_dataframe(df)
    preview = masked.head(MAX_GRID_ROWS)

    rows: list[dict[str, Any]] = []
    for _, row in preview.iterrows():
        rows.append({str(col): _to_jsonable_value(val) for col, val in row.items()})

    return {
        f"{prefix}_columns": [str(c) for c in masked.columns],
        f"{prefix}_rows": rows,
        f"{prefix}_row_count": int(len(masked)),
        f"{prefix}_truncated": bool(len(masked) > MAX_GRID_ROWS),
    }


def _build_session_grid_payload(entry: dict[str, Any]) -> dict[str, Any]:
    edited_df = entry.get("df")
    if not isinstance(edited_df, pd.DataFrame):
        raise HTTPException(status_code=500, detail="세션 데이터가 손상되었습니다.")

    original_df = entry.get("original_df")
    if not isinstance(original_df, pd.DataFrame):
        original_df = edited_df.copy(deep=True)
        entry["original_df"] = original_df

    payload: dict[str, Any] = {}
    payload.update(_build_grid_payload(edited_df, prefix="grid"))
    payload.update(_build_grid_payload(original_df, prefix="original_grid"))
    payload.update(_build_grid_payload(edited_df, prefix="edited_grid"))
    return payload


def _safe_filename(raw_name: str, default_stem: str = "datalens") -> str:
    name = (raw_name or "").strip()
    if not name:
        name = default_stem

    # Path separator/제어문자 제거
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._")
    return name or default_stem


def _content_disposition(filename: str) -> str:
    quoted = quote(filename)
    return f"attachment; filename*=UTF-8''{quoted}"


def _resolve_pdf_font() -> tuple[str, bool]:
    """
    Returns (font_name, supports_full_unicode).
    """
    from reportlab.pdfbase import pdfmetrics  # type: ignore[reportMissingImports]
    from reportlab.pdfbase.ttfonts import TTFont  # type: ignore[reportMissingImports]

    env_font_path = os.environ.get("DATALENS_PDF_FONT_PATH", "").strip()
    candidate_paths = [
        env_font_path,
        os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.ttf"),
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/malgun.ttf",
    ]

    for idx, path in enumerate(candidate_paths):
        if not path:
            continue
        if not os.path.exists(path):
            continue
        try:
            font_name = f"DLUnicodeFont{idx}"
            pdfmetrics.registerFont(TTFont(font_name, path))
            return font_name, True
        except Exception:
            continue

    # CJK CID 폰트는 외부 TTF 없이도 한글 출력이 가능한 경우가 많아 서버 배포 환경에서 유용함
    try:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont  # type: ignore[reportMissingImports]

        cid_name = "HYSMyeongJo-Medium"
        pdfmetrics.registerFont(UnicodeCIDFont(cid_name))
        return cid_name, True
    except Exception:
        return "Helvetica", False


def _build_report_pdf_bytes(report: dict[str, Any]) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4  # type: ignore[reportMissingImports]
        from reportlab.pdfgen import canvas  # type: ignore[reportMissingImports]
    except Exception:
        # reportlab 미설치 환경에서도 다운로드 자체는 가능해야 하므로 최소 PDF로 폴백
        return _build_minimal_pdf_bytes(report)

    preferred_font, full_unicode = _resolve_pdf_font()

    def _safe_text(text: Any) -> str:
        raw = str(text)
        if full_unicode:
            return raw
        # 폰트 로딩 실패 시 최소한 PDF 생성은 유지하되 문자는 손상될 수 있음
        return raw.encode("ascii", "replace").decode("ascii")

    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    insights = report.get("insights", []) if isinstance(report.get("insights"), list) else []
    table_data = report.get("table_data", []) if isinstance(report.get("table_data"), list) else []

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 48

    def _line(text: str, font: str | None = None, size: int = 10, step: int = 16) -> None:
        nonlocal y
        if y < 50:
            c.showPage()
            y = height - 48
        c.setFont(font or preferred_font, size)
        c.drawString(36, y, _safe_text(text))
        y -= step

    title_font = "Helvetica-Bold" if preferred_font == "Helvetica" else preferred_font
    _line("Data Lens Analysis Report", title_font, 14, 22)
    _line(f"Report ID: {report.get('id', '-')}")
    _line(f"File: {report.get('file_name', '-')}")
    _line(f"Created At: {report.get('created_at', '-')}")
    _line(f"Recommended Method: {report.get('recommended_method', '-')}")
    _line("")

    _line("Summary", title_font, 12, 18)
    _line(f"Rows: {summary.get('row_count', 0)}")
    _line(f"Columns: {summary.get('column_count', 0)}")
    _line(f"Missing Total: {summary.get('missing_total', 0)}")
    _line("")

    _line("Explanation", title_font, 12, 18)
    for chunk in _safe_text(report.get("explanation", "")).split("\n"):
        _line(chunk)
    _line("")

    _line("Insights", title_font, 12, 18)
    for idx, insight in enumerate(insights[:12], start=1):
        _line(f"{idx}. {insight}")
    _line("")

    _line("Top Columns", title_font, 12, 18)
    for col in table_data[:20]:
        if not isinstance(col, dict):
            continue
        _line(
            f"- {col.get('name', '-')}: dtype={col.get('dtype', '-')}, missing={col.get('missing', 0)}, unique={col.get('unique', '-')}, mean={col.get('mean', '-')}"
        )

    c.save()
    buf.seek(0)
    return buf.read()


def _build_minimal_pdf_bytes(report: dict[str, Any]) -> bytes:
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    insights = report.get("insights", []) if isinstance(report.get("insights"), list) else []

    lines = [
        "Data Lens Analysis Report",
        f"Report ID: {report.get('id', '-')}",
        f"File: {report.get('file_name', '-')}",
        f"Recommended Method: {report.get('recommended_method', '-')}",
        f"Rows: {summary.get('row_count', 0)}",
        f"Columns: {summary.get('column_count', 0)}",
        f"Missing Total: {summary.get('missing_total', 0)}",
        "",
        "Explanation:",
        str(report.get("explanation", "")),
        "",
        "Insights:",
    ]
    for idx, insight in enumerate(insights[:10], start=1):
        lines.append(f"{idx}. {insight}")

    def _to_ascii(text: str) -> str:
        return str(text).encode("ascii", "replace").decode("ascii")

    def _pdf_escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content_parts = ["BT", "/F1 10 Tf", "40 800 Td"]
    for line in lines:
        safe = _pdf_escape(_to_ascii(line))
        content_parts.append(f"({safe}) Tj")
        content_parts.append("0 -14 Td")
    content_parts.append("ET")
    stream_data = "\n".join(content_parts).encode("latin-1", errors="replace")

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        (
            f"5 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode("latin-1")
            + stream_data
            + b"\nendstream\nendobj\n"
        ),
    ]

    header = b"%PDF-1.4\n"
    body = b""
    offsets = [0]
    cursor = len(header)
    for obj in objects:
        offsets.append(cursor)
        body += obj
        cursor += len(obj)

    xref_start = len(header) + len(body)
    xref = f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    xref += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n".encode("latin-1")

    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    return header + body + xref + trailer


def _build_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="edited_data")
    output.seek(0)
    return output.read()


def _normalize_text_token(text: str) -> str:
    return re.sub(r"[\s_\-]", "", str(text).lower())


def _strip_particle(text: str) -> str:
    cleaned = text.strip()
    for suffix in ["입니다", "이야", "이", "가", "은", "는", "을", "를", "도", "값"]:
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return cleaned.strip()


def _resolve_column_name(col_hint: str, columns: list[Any]) -> str | None:
    if not col_hint:
        return None

    hint_norm = _normalize_text_token(_strip_particle(col_hint))
    if not hint_norm:
        return None

    exact = [str(c) for c in columns if _normalize_text_token(str(c)) == hint_norm]
    if exact:
        return exact[0]

    contains = [str(c) for c in columns if hint_norm in _normalize_text_token(str(c))]
    if contains:
        return contains[0]

    reverse_contains = [str(c) for c in columns if _normalize_text_token(str(c)) in hint_norm]
    if reverse_contains:
        return reverse_contains[0]

    return None


def _invert_op(op: str) -> str:
    return {
        "<": ">=",
        "<=": ">",
        ">": "<=",
        ">=": "<",
    }.get(op, op)


def _normalize_comp_op(op: str) -> str:
    mapping = {
        "이하": "<=",
        "미만": "<",
        "이상": ">=",
        "초과": ">",
        "<": "<",
        "<=": "<=",
        ">": ">",
        ">=": ">=",
    }
    return mapping.get(op.strip(), op.strip())


def _build_numeric_filter_code(column: str, op: str, value: str, exclude_mode: bool) -> str:
    use_op = _invert_op(op) if exclude_mode else op
    return (
        f"df = df[pd.to_numeric(df['{column}'], errors='coerce') {use_op} {value}]"
    )


def _build_random_fillna_code(command: str, df: pd.DataFrame) -> str | None:
    q = (command or "").strip()
    if not q:
        return None

    random_intent = bool(re.search(r"무작위|랜덤|random", q, flags=re.IGNORECASE))
    fill_intent = bool(re.search(r"채워|넣어|집어|대체|입력|보정", q))
    missing_intent = bool(re.search(r"없는|결측|빈값|누락", q))

    if not random_intent or not (fill_intent or missing_intent):
        return None

    col_hint_match = re.search(r"(?P<col>[가-힣A-Za-z0-9_\s]+?)\s*(?:가|이|의)?\s*(?:없는|결측|빈값|누락)", q)
    col_hint = _strip_particle(col_hint_match.group("col")) if col_hint_match else ""

    if "나이" in q and not col_hint:
        col_hint = "나이"

    resolved_col = _resolve_column_name(col_hint, list(df.columns)) if col_hint else None

    if not resolved_col:
        if "나이" in q:
            age_like_candidates: list[str] = []
            for c in df.columns:
                num = pd.to_numeric(df[c], errors="coerce")
                non_null = num.dropna()
                if len(non_null) == 0:
                    continue
                if int(num.isna().sum()) <= 0:
                    continue

                min_v = float(non_null.min())
                max_v = float(non_null.max())
                is_integer_like = bool(np.all(np.isclose(non_null.values, np.round(non_null.values))))

                # 일반적인 나이 분포(정수형, 0~120) 우선
                if is_integer_like and min_v >= 0 and max_v <= 120:
                    age_like_candidates.append(str(c))

            if age_like_candidates:
                resolved_col = age_like_candidates[0]

    if not resolved_col:
        candidates = []
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if int(s.isna().sum()) > 0:
                candidates.append(str(c))
        if candidates:
            age_like = [c for c in candidates if "나이" in c]
            resolved_col = age_like[0] if age_like else candidates[0]

    if not resolved_col:
        return None

    max_match = re.search(r"(?:최대|맥스|max)\s*(?P<max>\d+(?:\.\d+)?)", q, flags=re.IGNORECASE)
    if not max_match:
        max_match = re.search(r"(?P<max>\d+(?:\.\d+)?)\s*세\s*까지", q)
    min_match = re.search(r"(?:최소|min)\s*(?P<min>\d+(?:\.\d+)?)", q, flags=re.IGNORECASE)

    max_val = float(max_match.group("max")) if max_match else 60.0
    min_val = float(min_match.group("min")) if min_match else (1.0 if "나이" in resolved_col else 0.0)

    if max_val <= min_val:
        max_val = min_val + 1.0

    numeric_series = pd.to_numeric(df[resolved_col], errors="coerce")
    non_null = numeric_series.dropna()
    integer_like = bool("나이" in resolved_col)
    if len(non_null) > 0 and not integer_like:
        integer_like = bool(np.all(np.isclose(non_null.values, np.round(non_null.values))))

    if integer_like:
        low = int(np.floor(min_val))
        high_inclusive = int(np.floor(max_val))
        if high_inclusive <= low:
            high_inclusive = low + 1
        random_expr = f"np.random.randint({low}, {high_inclusive + 1}, size=df.shape[0])"
    else:
        random_expr = f"np.random.uniform({float(min_val):.6f}, {float(max_val):.6f}, size=df.shape[0])"

    return (
        "df = df.assign(**{" +
        f"'{resolved_col}': pd.to_numeric(df['{resolved_col}'], errors='coerce').where(pd.to_numeric(df['{resolved_col}'], errors='coerce').notna(), {random_expr})" +
        "})"
    )


def _guess_category_filter_code(token: str, df: pd.DataFrame) -> str | None:
    token = _strip_particle(token)
    if not token:
        return None

    synonyms = {
        "여성": "여",
        "남성": "남",
    }
    candidates = [token]
    if token in synonyms:
        candidates.append(synonyms[token])

    best: tuple[str, str] | None = None
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        uniques = [str(v).strip() for v in series.dropna().unique()[:200]]
        for cand in candidates:
            for uv in uniques:
                if not uv:
                    continue
                uv_norm = _normalize_text_token(uv)
                cand_norm = _normalize_text_token(cand)
                if cand_norm == uv_norm or cand_norm in uv_norm or uv_norm in cand_norm:
                    best = (str(col), uv)
                    break
            if best:
                break
        if best:
            break

    if not best:
        return None

    col, val = best
    return f"df = df[df['{col}'].astype('string').str.strip() == '{val}']"


def _extract_python_line(text: str) -> str:
    cleaned = re.sub(r"```(?:python)?", "", text).strip()
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("모델 응답에서 코드를 찾을 수 없습니다.")
    code = lines[0]
    if code.startswith("json") and len(lines) > 1:
        code = lines[1]
    return code.strip()


def _validate_edit_code(code: str) -> str:
    if not isinstance(code, str):
        raise ValueError("수정 코드는 문자열이어야 합니다.")

    cleaned = code.strip()
    if not cleaned:
        raise ValueError("빈 수정 코드는 허용되지 않습니다.")
    if "\n" in cleaned or ";" in cleaned:
        raise ValueError("수정 코드는 반드시 한 줄이어야 합니다.")
    if not cleaned.startswith("df ="):
        raise ValueError("수정 코드는 반드시 df에 재할당해야 합니다.")

    lowered = cleaned.lower()
    for pattern in _BANNED_CODE_PATTERNS:
        if re.search(pattern, lowered):
            raise ValueError("허용되지 않는 코드 패턴이 포함되어 있습니다.")

    try:
        tree = ast.parse(cleaned, mode="exec")
    except Exception as exc:
        raise ValueError(f"코드 구문 오류: {exc}") from exc

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        raise ValueError("단일 할당문만 허용됩니다.")

    assign_stmt = tree.body[0]
    if len(assign_stmt.targets) != 1 or not isinstance(assign_stmt.targets[0], ast.Name):
        raise ValueError("df 단일 재할당만 허용됩니다.")
    if assign_stmt.targets[0].id != "df":
        raise ValueError("좌변 변수는 반드시 df여야 합니다.")

    blocked_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.For,
        ast.While,
        ast.With,
        ast.Try,
        ast.Raise,
        ast.Lambda,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
    )

    for node in ast.walk(tree):
        if isinstance(node, blocked_nodes):
            raise ValueError("허용되지 않는 코드 구조가 포함되어 있습니다.")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAME_IDS:
            raise ValueError(f"허용되지 않은 식별자 사용: {node.id}")
        if isinstance(node, ast.Attribute) and str(node.attr).startswith("__"):
            raise ValueError("dunder 속성 접근은 허용되지 않습니다.")

    return cleaned


def _execute_edit_code(df: pd.DataFrame, code: str) -> pd.DataFrame:
    validated = _validate_edit_code(code)
    local_scope: dict[str, Any] = {
        "df": df.copy(deep=True),
        "pd": pd,
        "np": np,
    }
    try:
        exec(validated, {"__builtins__": {}}, local_scope)
    except Exception as exc:
        raise ValueError(f"코드 실행 실패: {exc}") from exc

    result_df = local_scope.get("df")
    if not isinstance(result_df, pd.DataFrame):
        raise ValueError("수정 코드 실행 결과가 DataFrame이 아닙니다.")

    return _clean_dataframe(result_df)


def _fallback_edit_code(command: str, df: pd.DataFrame) -> str:
    q = (command or "").strip()
    if not q:
        raise ValueError("수정 명령이 비어 있습니다.")

    random_fill = _build_random_fillna_code(q, df)
    if random_fill:
        return random_fill

    numeric_filter = re.search(
        r"(?P<col>[가-힣A-Za-z0-9_\s]+?)\s*(?P<op>이하|미만|이상|초과|<=|>=|<|>)\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?:인)?\s*(?:행|데이터|사람)?\s*(?P<action>제외|삭제|빼고|빼줘|제거|남겨|필터|유지)",
        q,
    )
    if numeric_filter:
        col_hint = _strip_particle(numeric_filter.group("col"))
        resolved_col = _resolve_column_name(col_hint, list(df.columns))
        if not resolved_col:
            raise ValueError(f"컬럼을 찾을 수 없습니다: {col_hint}")
        op = _normalize_comp_op(numeric_filter.group("op"))
        action = numeric_filter.group("action")
        exclude_mode = action in ["제외", "삭제", "빼고", "빼줘", "제거"]
        return _build_numeric_filter_code(resolved_col, op, numeric_filter.group("value"), exclude_mode)

    keep_category = re.search(
        r"(?P<token>[가-힣A-Za-z0-9_\s]+?)\s*(?:데이터)?\s*만\s*(?:남겨|유지|필터)",
        q,
    )
    if keep_category:
        guessed = _guess_category_filter_code(keep_category.group("token"), df)
        if guessed:
            return guessed

    set_all = re.search(
        r"(?P<col>[가-힣A-Za-z0-9_\s]+?)\s*(?:값|수치)?\s*(?:을|를)?\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?:으로|로)\s*(?:바꿔|변경|수정)",
        q,
    )
    if set_all:
        col_hint = _strip_particle(set_all.group("col"))
        resolved_col = _resolve_column_name(col_hint, list(df.columns))
        if not resolved_col:
            raise ValueError(f"컬럼을 찾을 수 없습니다: {col_hint}")
        return f"df = df.assign(**{{'{resolved_col}': {set_all.group('value')}}})"

    fill_na = re.search(
        r"(?P<col>[가-힣A-Za-z0-9_\s]+?)\s*(?:결측치|빈값|누락값)\s*(?:을|를)?\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?:으로|로)\s*(?:채워|대체|변경)",
        q,
    )
    if fill_na:
        col_hint = _strip_particle(fill_na.group("col"))
        resolved_col = _resolve_column_name(col_hint, list(df.columns))
        if not resolved_col:
            raise ValueError(f"컬럼을 찾을 수 없습니다: {col_hint}")
        val = fill_na.group("value")
        return (
            "df = df.assign(**{" +
            f"'{resolved_col}': pd.to_numeric(df['{resolved_col}'], errors='coerce').fillna({val})" +
            "})"
        )

    raise ValueError(
        "명령을 자동 해석하지 못했습니다. 예: '소득이 3000 이하인 행 제외', '여성 데이터만 남겨', '나이 결측치를 0으로 채워'."
    )


def _generate_edit_code(command: str, df: pd.DataFrame) -> str:
    use_llm = os.environ.get("DATALENS_USE_LLM_EDIT", os.environ.get("DATALENS_USE_LLM", "0")) == "1"

    if use_llm:
        try:
            schema = [
                {
                    "name": str(col),
                    "dtype": str(df[col].dtype),
                    "samples": [_to_jsonable_value(v) for v in df[col].dropna().head(3).tolist()],
                }
                for col in df.columns
            ]
            llm_response = ollama.chat(
                model="llama3.2",
                messages=[
                    {"role": "system", "content": EDIT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "user_command": command,
                                "columns": schema,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            )
            candidate = _extract_python_line(llm_response["message"]["content"])
            return _validate_edit_code(candidate)
        except Exception:
            pass

    return _fallback_edit_code(command, df)


def _is_id_like_column(col_name: str, series: pd.Series) -> bool:
    name = col_name.lower()
    if any(k in name for k in ["id", "idx", "code", "코드", "번호"]):
        return True
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    # 소표본에서는 고유비율이 높아도 연속형 지표일 가능성이 커서 ID로 보지 않음
    if len(non_null) < 30:
        return False

    if pd.api.types.is_numeric_dtype(non_null):
        # 숫자형인데 소수 성분이 뚜렷하면 연속형 지표로 간주
        as_float = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(as_float) == 0:
            return False
        frac = (as_float - np.floor(as_float)).abs().mean()
        if frac > 1e-6:
            return False

    unique_ratio = non_null.nunique() / len(non_null)
    return unique_ratio > 0.98


def _is_categorical_candidate(col_name: str, series: pd.Series) -> bool:
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    nun = int(non_null.nunique())
    if nun < 2:
        return False

    if _is_id_like_column(str(col_name), series):
        return False

    # 범주형 후보는 수준 수가 과도하지 않아야 함 (ID/고유키 열 제외)
    max_levels = min(20, max(6, int(len(non_null) * 0.2)))
    if nun > max_levels:
        return False

    unique_ratio = nun / max(len(non_null), 1)
    if unique_ratio > 0.7:
        return False

    return True


def _build_numeric_profile_chart_data(df: pd.DataFrame) -> list[dict[str, Any]]:
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not _is_id_like_column(str(c), df[c])
    ]

    chart_data_raw = []
    for col in numeric_cols[:5]:
        non_null = df[col].dropna()
        if len(non_null) > 0:
            chart_data_raw.append({"category": str(col), "raw_mean": float(non_null.mean())})

    chart_data = []
    if chart_data_raw:
        means = [item["raw_mean"] for item in chart_data_raw]
        mn, mx = min(means), max(means)
        for item in chart_data_raw:
            if mx > mn:
                scaled = (item["raw_mean"] - mn) / (mx - mn) * 100
            else:
                scaled = 50.0
            chart_data.append(
                {
                    "category": item["category"],
                    "value": round(float(scaled), 2),
                    "raw_mean": round(float(item["raw_mean"]), 3),
                }
            )
    return chart_data


def _build_group_metric_chart_data(df: pd.DataFrame, evidence: dict[str, Any]) -> dict[str, Any] | None:
    preferred_pairs = []
    if isinstance(evidence.get("anova"), dict):
        preferred_pairs.append((evidence["anova"].get("group_col"), evidence["anova"].get("value_col")))
    if isinstance(evidence.get("t_test"), dict):
        preferred_pairs.append((evidence["t_test"].get("group_col"), evidence["t_test"].get("value_col")))

    # 근거에서 집단/지표 쌍을 못 찾은 경우를 대비한 자동 탐색
    if not preferred_pairs:
        categorical_cols = []
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                if s.dropna().nunique() <= 12 and _is_categorical_candidate(str(c), s):
                    categorical_cols.append(c)
            else:
                if _is_categorical_candidate(str(c), s):
                    categorical_cols.append(c)

        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not _is_id_like_column(str(c), df[c])
        ]

        if categorical_cols and numeric_cols:
            group_name_keys = ["group", "그룹", "구분", "등급", "class", "cluster"]

            def _group_col_score(col: Any) -> float:
                s = df[col]
                non_null = s.dropna()
                nun = non_null.nunique()
                if nun < 2 or nun > 12:
                    return -1.0

                score = 0.0
                lowered = str(col).lower()
                if any(k in lowered for k in group_name_keys):
                    score += 10.0

                # 범주 수준이 너무 많지 않고, 결측이 적을수록 우선
                score += (1.0 - abs(4 - min(nun, 8)) / 8.0) * 2.0
                score += (len(non_null) / max(len(df), 1)) * 2.0
                return score

            best_group = max(categorical_cols, key=_group_col_score)

            metric_priority_keys = ["target", "타겟", "score", "점수", "매출", "sales", "revenue", "소득", "금액"]

            prioritized_numeric = []
            for key in metric_priority_keys:
                for c in numeric_cols:
                    if key in str(c).lower() and c not in prioritized_numeric:
                        prioritized_numeric.append(c)

            def _value_col_score(col: Any) -> float:
                lowered = str(col).lower()
                non_null = df[col].dropna()
                if len(non_null) == 0:
                    return -1.0
                score = 0.0
                if any(k in lowered for k in metric_priority_keys):
                    score += 10.0
                score += float(non_null.std()) / (abs(float(non_null.mean())) + 1.0)
                score += len(non_null) / max(len(df), 1)
                return score

            if prioritized_numeric:
                best_value = prioritized_numeric[0]
            else:
                best_value = max(numeric_cols, key=_value_col_score)
            preferred_pairs.append((best_group, best_value))

    for gcol, vcol in preferred_pairs:
        if not gcol or not vcol:
            continue
        if gcol not in df.columns or vcol not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[vcol]):
            continue

        sub = df[[gcol, vcol]].copy()
        sub[gcol] = sub[gcol].astype("string").str.strip()
        sub[gcol] = sub[gcol].replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
        sub = sub.dropna(subset=[gcol, vcol])
        if len(sub) == 0:
            continue

        group_n = sub[gcol].nunique()
        if group_n < 2 or group_n > 12:
            continue

        grouped = sub.groupby(gcol, as_index=False)[vcol].mean().rename(columns={vcol: "raw_mean"})
        if len(grouped) < 2:
            continue

        grouped = grouped.sort_values("raw_mean", ascending=False)
        mn, mx = float(grouped["raw_mean"].min()), float(grouped["raw_mean"].max())

        rows: list[dict[str, Any]] = []
        group_count = len(grouped)
        for _, row in grouped.iterrows():
            raw_mean = float(row["raw_mean"])
            # 2집단 비교에서 min-max 스케일은 0/100 극단값을 만들기 쉬워 상대비율 스케일 우선 사용
            if group_count <= 2 and mx > 0 and mn >= 0:
                scaled = (raw_mean / mx) * 100
            else:
                scaled = ((raw_mean - mn) / (mx - mn) * 100) if mx > mn else 50.0
            rows.append(
                {
                    "category": str(row[gcol]),
                    "value": round(float(scaled), 2),
                    "raw_mean": round(raw_mean, 3),
                }
            )
        return {
            "chart_data": rows,
            "group_col": str(gcol),
            "value_col": str(vcol),
            "n_groups": int(group_count),
        }

    return None


def _read_upload(file: UploadFile) -> pd.DataFrame:
    filename = (file.filename or "").lower()
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    try:
        if filename.endswith(".csv"):
            parsed = pd.read_csv(BytesIO(raw))
            return _clean_dataframe(parsed)
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            parsed = pd.read_excel(BytesIO(raw))
            return _clean_dataframe(parsed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"파일 파싱 실패: {exc}") from exc

    raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다. CSV/XLSX만 지원합니다.")


def _build_summary(df: pd.DataFrame) -> dict[str, Any]:
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())

    column_summaries = []
    for col in df.columns:
        s = df[col]
        info: dict[str, Any] = {
            "name": str(col),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            non_null = s.dropna()
            if len(non_null) > 0:
                info["mean"] = round(float(non_null.mean()), 4)
                info["min"] = round(float(non_null.min()), 4)
                info["max"] = round(float(non_null.max()), 4)
        column_summaries.append(info)

    chart_data = _build_numeric_profile_chart_data(df)

    return {
        "row_count": int(rows),
        "column_count": int(cols),
        "missing_total": missing,
        "columns": column_summaries,
        "chart_data": chart_data,
    }


def _compute_stat_evidence(df: pd.DataFrame) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "t_test": None,
        "correlation": None,
        "anova": None,
        "regression": None,
        "chi_square": None,
    }

    # 숫자형 중 연속형 후보만 사용(고유값이 너무 적은 컬럼 제외)
    numeric_cols = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if _is_id_like_column(str(c), df[c]):
            continue
        if df[c].dropna().nunique() <= 8:
            continue
        numeric_cols.append(c)

    categorical_cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            if s.dropna().nunique() <= 8 and _is_categorical_candidate(str(c), s):
                categorical_cols.append(c)
        else:
            if _is_categorical_candidate(str(c), s):
                categorical_cols.append(c)

    # 1) T-검정: 이진 집단 + 수치형 (효과크기 d가 큰 조합)
    best_t = None
    best_d = -1.0
    for gcol in categorical_cols:
        non_null = df[gcol].dropna()
        if non_null.nunique() != 2:
            continue
        levels = list(non_null.unique())
        for vcol in numeric_cols:
            subset = df[[gcol, vcol]].dropna()
            if len(subset) < 30:
                continue
            g1 = subset[subset[gcol] == levels[0]][vcol]
            g2 = subset[subset[gcol] == levels[1]][vcol]
            if len(g1) < 10 or len(g2) < 10:
                continue
            t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
            pooled_std = np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)
            d = abs(float((g1.mean() - g2.mean()) / pooled_std)) if pooled_std > 0 else 0.0
            if d > best_d:
                best_d = d
                best_t = {
                    "group_col": str(gcol),
                    "value_col": str(vcol),
                    "group_a": str(levels[0]),
                    "group_b": str(levels[1]),
                    "mean_a": round(float(g1.mean()), 3),
                    "mean_b": round(float(g2.mean()), 3),
                    "t_stat": round(float(t_stat), 3),
                    "p_value": round(float(p_val), 4),
                    "cohen_d": round(float(d), 3),
                }
    evidence["t_test"] = best_t

    # 2) 상관분석: 절대 상관계수 최대쌍
    if len(numeric_cols) >= 2:
        best_pair = None
        best_abs_r = -1.0
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                subset = df[[c1, c2]].dropna()
                if len(subset) < 30:
                    continue
                r, p_val = stats.pearsonr(subset[c1], subset[c2])
                if abs(float(r)) > best_abs_r:
                    best_abs_r = abs(float(r))
                    best_pair = {
                        "col_x": str(c1),
                        "col_y": str(c2),
                        "r": round(float(r), 3),
                        "p_value": round(float(p_val), 4),
                    }
        evidence["correlation"] = best_pair

    # 3) ANOVA: 3개 이상 집단 + 수치형 (eta^2 최대 조합)
    best_anova = None
    best_eta = -1.0
    for gcol in categorical_cols:
        nun = df[gcol].dropna().nunique()
        if nun < 3 or nun > 12:
            continue
        for vcol in numeric_cols:
            sub = df[[gcol, vcol]].dropna()
            if len(sub) < 50:
                continue
            groups = [grp[vcol].values for _, grp in sub.groupby(gcol)]
            if len(groups) < 3 or any(len(g) < 10 for g in groups):
                continue
            f_stat, p_val = stats.f_oneway(*groups)
            all_vals = np.concatenate(groups)
            grand = all_vals.mean()
            ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
            ss_total = sum((all_vals - grand) ** 2)
            eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
            if eta_sq > best_eta:
                best_eta = eta_sq
                best_anova = {
                    "group_col": str(gcol),
                    "value_col": str(vcol),
                    "f_stat": round(float(f_stat), 3),
                    "p_value": round(float(p_val), 4),
                    "eta_squared": round(float(eta_sq), 3),
                    "n_groups": int(nun),
                }
    evidence["anova"] = best_anova

    # 4) 회귀분석: 구매/매출 계열 컬럼 우선 타깃 선택
    if len(numeric_cols) >= 3:
        target = None
        priority_keys = ["매출", "구매", "revenue", "sales", "amount", "금액", "총구매"]
        for c in numeric_cols:
            lowered = str(c).lower()
            if any(k.lower() in lowered for k in priority_keys):
                target = c
                break
        if target is None:
            target = max(numeric_cols, key=lambda c: float(df[c].dropna().std()) if len(df[c].dropna()) else 0)

        corr_scores = []
        for c in numeric_cols:
            if c == target:
                continue
            sub = df[[target, c]].dropna()
            if len(sub) < 30:
                continue
            r, _ = stats.pearsonr(sub[target], sub[c])
            corr_scores.append((abs(float(r)), c))
        corr_scores.sort(reverse=True)
        features = [c for _, c in corr_scores[:3]]
        if len(features) >= 2:
            evidence["regression"] = {
                "target": str(target),
                "features": [str(f) for f in features],
                "note": "다중회귀 후보",
            }

    # 5) 카이제곱: 범주형-범주형 조합
    best_chi = None
    best_chi2 = -1.0
    if len(categorical_cols) >= 2:
        for i in range(len(categorical_cols)):
            for j in range(i + 1, len(categorical_cols)):
                c1, c2 = categorical_cols[i], categorical_cols[j]
                sub = df[[c1, c2]].dropna()
                if len(sub) < 50:
                    continue
                tbl = pd.crosstab(sub[c1], sub[c2])
                if tbl.shape[0] < 2 or tbl.shape[1] < 2:
                    continue
                chi2, p_val, dof, _ = stats.chi2_contingency(tbl)
                if chi2 > best_chi2:
                    best_chi2 = float(chi2)
                    best_chi = {
                        "col1": str(c1),
                        "col2": str(c2),
                        "chi2": round(float(chi2), 3),
                        "p_value": round(float(p_val), 4),
                        "dof": int(dof),
                    }
        evidence["chi_square"] = best_chi

    return evidence


def _normalize_method_name(method: str) -> str:
    m = (method or "").lower()
    if "anova" in m or "분산분석" in m:
        return "ANOVA"
    if "t-검정" in m or "t 검정" in m or "ttest" in m or "t-test" in m:
        return "T-검정"
    if "회귀" in m or "regression" in m or "predict" in m or "예측" in m:
        return "회귀분석"
    if "상관" in m or "correlation" in m or "관계" in m:
        return "상관분석"
    if "chi" in m or "카이" in m or "교차" in m:
        return "교차분석"
    return method or "기술통계"


def _compute_method_scores(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}

    corr = evidence.get("correlation")
    if corr:
        p_bonus = 15 if corr.get("p_value", 1) < 0.05 else 0
        base = min(100.0, abs(float(corr.get("r", 0))) * 100 + p_bonus)
        scores["상관분석"] = {
            "score": round(base, 1),
            "reason": f"r={corr.get('r')}, p={corr.get('p_value')}",
        }
    else:
        scores["상관분석"] = {"score": 0.0, "reason": "유효한 수치형 변수쌍 부족"}

    reg = evidence.get("regression")
    if reg:
        n_feat = len(reg.get("features", []))
        base = min(100.0, 45 + n_feat * 15)
        scores["회귀분석"] = {
            "score": round(base, 1),
            "reason": f"target={reg.get('target')}, features={', '.join(reg.get('features', []))}",
        }
    else:
        scores["회귀분석"] = {"score": 0.0, "reason": "타깃/설명변수 후보 부족"}

    anova = evidence.get("anova")
    if anova:
        p_bonus = 15 if anova.get("p_value", 1) < 0.05 else 0
        base = min(100.0, float(anova.get("eta_squared", 0)) * 100 + p_bonus)
        scores["ANOVA"] = {
            "score": round(base, 1),
            "reason": f"eta²={anova.get('eta_squared')}, p={anova.get('p_value')}",
        }
    else:
        scores["ANOVA"] = {"score": 0.0, "reason": "3개 이상 유효 집단 비교 근거 부족"}

    t_e = evidence.get("t_test")
    if t_e:
        p_bonus = 15 if t_e.get("p_value", 1) < 0.05 else 0
        base = min(100.0, abs(float(t_e.get("cohen_d", 0))) * 100 + p_bonus)
        scores["T-검정"] = {
            "score": round(base, 1),
            "reason": f"d={t_e.get('cohen_d')}, p={t_e.get('p_value')}",
        }
    else:
        scores["T-검정"] = {"score": 0.0, "reason": "2개 집단 비교 근거 부족"}

    chi = evidence.get("chi_square")
    if chi:
        # effect size가 없어도 p-value와 유효 테이블 존재 여부로 기본 점수 부여
        p_bonus = 15 if chi.get("p_value", 1) < 0.05 else 5
        base = min(100.0, 45 + p_bonus)
        scores["교차분석"] = {
            "score": round(base, 1),
            "reason": f"chi2={chi.get('chi2')}, p={chi.get('p_value')}",
        }
    else:
        scores["교차분석"] = {"score": 0.0, "reason": "유효한 범주형 변수쌍 부족"}

    return scores


def _build_method_visualization(
    df: pd.DataFrame,
    evidence: dict[str, Any],
    method: str,
    default_chart_data: list[dict[str, Any]],
) -> dict[str, Any]:
    method_norm = _normalize_method_name(method)

    def _with_fallback(meta_title: str, meta_desc: str) -> dict[str, Any]:
        return {
            "chart_data": default_chart_data,
            "secondary_chart_data": [],
            "chart_meta": {
                "mode": "numeric_profile",
                "primary_chart": "bar",
                "secondary_chart": "line",
                "title": meta_title,
                "secondary_title": "추세 분석",
                "description": meta_desc,
                "value_key": "value",
            },
        }

    if method_norm in ["ANOVA", "T-검정"] and isinstance(evidence.get("group_metric"), dict):
        gm = evidence.get("group_metric")
        return {
            "chart_data": default_chart_data,
            "secondary_chart_data": [
                {
                    "category": str(r.get("category")),
                    "value": float(r.get("raw_mean", r.get("value", 0))),
                }
                for r in default_chart_data
            ],
            "chart_meta": {
                "mode": "group_compare",
                "primary_chart": "bar",
                "secondary_chart": "line",
                "title": f"{gm.get('group_col')}별 {gm.get('value_col')} 평균 비교",
                "secondary_title": "그룹 평균 추세",
                "description": f"{gm.get('group_col')} 집단별 {gm.get('value_col')} 평균을 비교합니다.",
                "value_key": "raw_mean",
            },
        }

    if method_norm == "상관분석" and isinstance(evidence.get("correlation"), dict):
        corr = evidence.get("correlation")
        x_col = corr.get("col_x")
        y_col = corr.get("col_y")
        if x_col in df.columns and y_col in df.columns:
            sub = df[[x_col, y_col]].dropna()
            if len(sub) >= 30:
                n_bins = min(8, max(4, int(len(sub) ** 0.5 // 2)))
                try:
                    sub = sub.copy()
                    sub["_bin"] = pd.qcut(sub[x_col], q=n_bins, duplicates="drop")
                    grouped = sub.groupby("_bin", observed=True)[y_col].mean().reset_index()
                    grouped = grouped.rename(columns={y_col: "raw_mean"})
                    rows = []
                    for i, row in grouped.iterrows():
                        raw = float(row["raw_mean"])
                        rows.append({"category": f"Q{i+1}", "value": round(raw, 3), "raw_mean": round(raw, 3)})
                    return {
                        "chart_data": rows,
                        "secondary_chart_data": rows,
                        "chart_meta": {
                            "mode": "correlation_curve",
                            "primary_chart": "line",
                            "secondary_chart": "bar",
                            "title": f"{x_col} 구간별 {y_col} 평균 추세",
                            "secondary_title": "구간 평균 비교",
                            "description": f"상관 강도(r={corr.get('r')})가 큰 변수쌍의 구간별 평균 패턴입니다.",
                            "value_key": "raw_mean",
                        },
                    }
                except Exception:
                    pass

    if method_norm == "회귀분석" and isinstance(evidence.get("regression"), dict):
        reg = evidence.get("regression")
        target = reg.get("target")
        features = reg.get("features", [])
        if target in df.columns and isinstance(features, list) and features:
            rows = []
            for f in features:
                if f not in df.columns:
                    continue
                sub = df[[target, f]].dropna()
                if len(sub) < 30:
                    continue
                r, _ = stats.pearsonr(sub[target], sub[f])
                strength = abs(float(r))
                rows.append(
                    {
                        "category": str(f),
                        "value": round(strength * 100, 2),
                        "raw_mean": round(strength, 3),
                    }
                )

            if rows:
                rows = sorted(rows, key=lambda x: x["value"], reverse=True)
                return {
                    "chart_data": rows,
                    "secondary_chart_data": rows,
                    "chart_meta": {
                        "mode": "regression_importance",
                        "primary_chart": "bar",
                        "secondary_chart": "line",
                        "title": f"{target} 예측 영향도(상관강도 기반)",
                        "secondary_title": "영향도 추세",
                        "description": "설명변수별 타깃 연관 강도(절대 상관계수)를 기준으로 정렬했습니다.",
                        "value_key": "value",
                    },
                }

    if method_norm == "교차분석" and isinstance(evidence.get("chi_square"), dict):
        chi = evidence.get("chi_square")
        c1, c2 = chi.get("col1"), chi.get("col2")
        if c1 in df.columns and c2 in df.columns:
            sub = df[[c1, c2]].dropna()
            if len(sub) >= 30:
                tbl = pd.crosstab(sub[c1], sub[c2], normalize="index") * 100
                if not tbl.empty:
                    main_col = tbl.sum(axis=0).idxmax()
                    rows = [
                        {
                            "category": str(idx),
                            "value": round(float(tbl.loc[idx, main_col]), 2),
                            "raw_mean": round(float(tbl.loc[idx, main_col]), 2),
                        }
                        for idx in tbl.index
                    ]
                    secondary = [
                        {
                            "category": str(col),
                            "value": round(float(tbl[col].mean()), 2),
                        }
                        for col in tbl.columns
                    ]
                    return {
                        "chart_data": rows,
                        "secondary_chart_data": secondary,
                        "chart_meta": {
                            "mode": "categorical_association",
                            "primary_chart": "bar",
                            "secondary_chart": "bar",
                            "title": f"{c1}별 {c2} 비율 비교",
                            "secondary_title": f"{c2} 평균 비율",
                            "description": f"교차분석의 기준 변수는 {c1}, 비교 변수는 {c2}입니다.",
                            "value_key": "value",
                        },
                    }

    return _with_fallback("변수 평균 비교", "데이터 특성상 기본 요약 시각화를 사용합니다.")


def _build_method_options(evidence: dict[str, Any]) -> list[dict[str, str]]:
    options = []

    scores = _compute_method_scores(evidence)

    corr = evidence.get("correlation")
    reg = evidence.get("regression")
    anova = evidence.get("anova")
    chi = evidence.get("chi_square")

    options.append(
        {
            "method": "상관분석 (Correlation)",
            "when_to_use": "두 수치형 변수 간 관계를 확인할 때",
            "current_fit": (
                f"점수 {scores['상관분석']['score']}/100: {corr['col_x']}–{corr['col_y']} (r={corr['r']}, p={corr['p_value']})"
                if corr else "적합도 중간: 수치형 변수가 충분하면 적용 가능"
            ),
        }
    )
    options.append(
        {
            "method": "회귀분석 (Regression)",
            "when_to_use": "여러 요인이 목표 변수에 미치는 영향을 보고 예측할 때",
            "current_fit": (
                f"점수 {scores['회귀분석']['score']}/100: target={reg['target']}, features={', '.join(reg['features'])}"
                if reg else "적합도 중간: 수치형 변수 3개 이상이면 권장"
            ),
        }
    )
    options.append(
        {
            "method": "분산분석 (ANOVA)",
            "when_to_use": "3개 이상 집단의 평균 차이를 비교할 때",
            "current_fit": (
                f"점수 {scores['ANOVA']['score']}/100: {anova['group_col']} vs {anova['value_col']} (p={anova['p_value']})"
                if anova else "적합도 중간: 3개 이상 집단 범주형 변수가 있으면 적용 가능"
            ),
        }
    )
    options.append(
        {
            "method": "교차분석 (Chi-square)",
            "when_to_use": "범주형 변수 간 관련성을 볼 때",
            "current_fit": (
                f"점수 {scores['교차분석']['score']}/100: {chi['col1']} vs {chi['col2']} (p={chi['p_value']})"
                if chi else "적합도 중간: 범주형 변수 2개 이상이면 적용 가능"
            ),
        }
    )

    method_key = {
        "상관분석 (Correlation)": "상관분석",
        "회귀분석 (Regression)": "회귀분석",
        "분산분석 (ANOVA)": "ANOVA",
        "교차분석 (Chi-square)": "교차분석",
    }
    options.sort(key=lambda x: float(scores[method_key[x["method"]]]["score"]), reverse=True)

    return options


def _choose_primary_method(question: str, evidence: dict[str, Any]) -> str:
    q = (question or "").lower()

    # 사용자가 기법을 명시하면 우선 존중
    if any(k in q for k in ["anova", "분산분석"]):
        return "ANOVA"
    if any(k in q for k in ["t-검정", "t 검정", "ttest", "t-test"]):
        return "T-검정"
    if any(k in q for k in ["chi", "카이", "교차분석", "카이제곱"]):
        return "교차분석"

    if any(k in q for k in ["예측", "영향", "회귀", "predict"]):
        return "회귀분석" if evidence.get("regression") else "상관분석"
    if any(k in q for k in ["관계", "상관", "correlation"]):
        return "상관분석" if evidence.get("correlation") else "회귀분석"
    if any(k in q for k in ["집단", "차이", "비교", "t-검정", "anova"]):
        if evidence.get("anova") and evidence["anova"]["p_value"] < 0.05:
            return "ANOVA"
        if evidence.get("t_test"):
            return "T-검정"

    # 기본: 근거 점수 최댓값
    scores = _compute_method_scores(evidence)
    ranked = sorted(
        [(k, float(v.get("score", 0))) for k, v in scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    if ranked and ranked[0][1] > 0:
        return ranked[0][0]
    return "기술통계"


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("모델 응답에서 JSON을 찾을 수 없습니다.")
    return json.loads(match.group(0))


def _safe_json_load(text: str, default_value: Any):
    try:
        return json.loads(text)
    except Exception:
        return default_value


def _serialize_report(model: AnalysisReportModel) -> dict[str, Any]:
    insights = _safe_json_load(model.insights_json, [])
    chart_data = _safe_json_load(model.chart_data_json, [])
    table_data = _safe_json_load(model.table_data_json, [])
    summary = _safe_json_load(model.summary_json, {})
    evidence = _safe_json_load(model.evidence_json, {})
    chart_meta = evidence.get("visualization") if isinstance(evidence, dict) else {}
    secondary_chart_data = evidence.get("secondary_chart_data") if isinstance(evidence, dict) else []
    return {
        "id": model.id,
        "file_name": model.file_name,
        "question": model.question,
        "recommended_method": model.recommended_method,
        "explanation": model.explanation,
        "insights": insights,
        "chart_data": chart_data,
        "table_data": table_data,
        "summary": summary,
        "evidence": evidence,
        "chart_meta": chart_meta if isinstance(chart_meta, dict) else {},
        "secondary_chart_data": secondary_chart_data if isinstance(secondary_chart_data, list) else [],
        "created_at": model.created_at.isoformat(),
    }


def _assistant_fallback_answer(report: dict[str, Any], question: str) -> str:
    rec = report.get("recommended_method", "기술통계")
    exp = report.get("explanation", "")
    insights = report.get("insights", [])
    summary = report.get("summary", {})
    evidence = report.get("evidence", {})
    chart_data = report.get("chart_data", [])
    q = (question or "").lower()

    if isinstance(evidence, dict) and isinstance(evidence.get("group_metric"), dict) and isinstance(chart_data, list) and chart_data:
        gm = evidence.get("group_metric")
        group_col = gm.get("group_col")
        value_col = gm.get("value_col")

        rows = []
        for r in chart_data:
            if isinstance(r, dict) and "category" in r and "raw_mean" in r:
                rows.append((str(r.get("category")), float(r.get("raw_mean"))))

        if rows and any(k in q for k in ["성별", "남성", "여성", "그룹", "집단", "평균", "비교"]):
            rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)
            top_label, top_mean = rows_sorted[0]
            bot_label, bot_mean = rows_sorted[-1]
            diff = top_mean - bot_mean

            male = next((v for k, v in rows if "남" in k), None)
            female = next((v for k, v in rows if "여" in k), None)

            lines = [
                f"현재 리포트에서 그룹 기준은 {group_col}, 비교 지표는 {value_col}입니다.",
                "그룹별 평균은 " + ", ".join([f"{k}={round(v, 3)}" for k, v in rows_sorted]) + " 입니다.",
                f"최고 그룹은 {top_label}, 최저 그룹은 {bot_label}이며 평균 차이는 {round(diff, 3)}입니다.",
            ]
            if male is not None and female is not None:
                sex_gap = female - male
                direction = "여성이 더 높고" if sex_gap > 0 else "남성이 더 높고"
                lines.append(f"성별 기준으로는 {direction} 평균 격차는 {round(abs(sex_gap), 3)}입니다.")
            lines.append("원하면 같은 기준으로 t-검정/ANOVA 유의성까지 바로 해석해 드릴게요.")
            return " ".join(lines)

    lines = [
        f"이번 분석의 핵심 기법은 {rec}입니다.",
        f"데이터 규모는 {summary.get('row_count', 0)}행, {summary.get('column_count', 0)}열이며 결측치는 {summary.get('missing_total', 0)}개입니다.",
    ]
    if exp:
        lines.append(f"주요 해석은 다음과 같습니다: {exp}")
    if isinstance(insights, list) and insights:
        lines.append(f"핵심 시사점은 {insights[0]}")
    if isinstance(evidence, dict) and evidence.get("correlation"):
        corr = evidence.get("correlation")
        lines.append(
            f"참고로 상관 근거는 {corr.get('col_x')}–{corr.get('col_y')}에서 r={corr.get('r')}, p={corr.get('p_value')}입니다."
        )
    lines.append(f"질문 '{question}'에 대해 더 구체적으로 원하면 어떤 지표를 기준으로 볼지 지정해 주세요.")
    return " ".join(lines)


def _has_foreign_script(text: str) -> bool:
    return bool(re.search(r"[A-Za-z\u3040-\u30FF\u4E00-\u9FFF]", text))


def _build_fallback_response(summary: dict[str, Any], evidence: dict[str, Any], question: str = "") -> dict[str, Any]:
    recommended = _choose_primary_method(question, evidence)
    method_options = _build_method_options(evidence)

    c_e = evidence.get("correlation")
    r_e = evidence.get("regression")
    a_e = evidence.get("anova")
    t_e = evidence.get("t_test")

    if recommended == "상관분석" and c_e:
        explain = (
            f"핵심 분석으로 상관분석을 권장합니다. {c_e['col_x']}와 {c_e['col_y']}의 상관계수는 "
            f"r={c_e['r']}, p={c_e['p_value']}로 확인되었습니다. "
            "변수 간 선형 관계를 빠르게 파악하고 후속 모델링의 방향을 정하는 데 유리합니다."
        )
    elif recommended == "회귀분석" and r_e:
        explain = (
            f"핵심 분석으로 회귀분석을 권장합니다. 목표변수는 {r_e['target']}이며, "
            f"주요 설명변수는 {', '.join(r_e['features'])}입니다. "
            "각 변수의 영향력을 동시에 확인하고 예측 모델을 구축할 수 있습니다."
        )
    elif recommended == "ANOVA" and a_e:
        explain = (
            f"핵심 분석으로 ANOVA를 권장합니다. {a_e['group_col']} 집단 간 {a_e['value_col']} 평균 차이를 비교했을 때 "
            f"p={a_e['p_value']}로 확인되었습니다. 3개 이상 집단 비교에 적합합니다."
        )
    elif recommended == "T-검정" and t_e:
        sig = "유의미" if t_e["p_value"] < 0.05 else "유의미하지 않음"
        explain = (
            f"핵심 분석으로 T-검정을 권장합니다. {t_e['group_col']}에 따른 {t_e['value_col']} 차이의 p값은 "
            f"{t_e['p_value']}로 {sig}합니다. 2개 집단 비교 상황에서 해석이 명확합니다."
        )
    else:
        explain = "현재 데이터는 기술통계를 먼저 확인한 뒤 상관/회귀/집단비교 분석으로 확장하는 것이 적절합니다."

    insights = [
        f"주 분석 추천: {recommended}",
        "대안 분석 3개를 함께 제시했으니 목적(관계 파악/영향 추정/집단 비교)에 따라 선택할 수 있습니다.",
        "원하면 지금 바로 대안 분석(상관/회귀/ANOVA/교차분석) 중 하나를 추가 실행할 수 있습니다.",
    ]

    return {
        "recommended_method": recommended,
        "explanation": explain,
        "insights": insights,
        "chart_data": summary.get("chart_data", []),
        "method_options": method_options,
        "next_question": "추가로 상관분석, 회귀분석, ANOVA, 교차분석 중 어떤 분석을 먼저 실행할까요?",
    }


def _call_llama(question: str, summary: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    user_prompt = (
        "[사용자 질문]\n"
        f"{question}\n\n"
        "[데이터 요약]\n"
        f"{json.dumps(summary, ensure_ascii=False)}\n\n"
        "[통계 근거]\n"
        f"{json.dumps(evidence, ensure_ascii=False)}"
    )

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response["message"]["content"]
    try:
        parsed = _extract_json(content)
    except Exception:
        return _build_fallback_response(summary, evidence, question)

    if _has_foreign_script(parsed.get("explanation", "")):
        # 한국어 교정 1회 시도
        rewrite = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": "너는 한국어 교정기다. 입력 JSON 스키마를 유지하고 explanation/insights를 한국어만 사용해 자연스럽게 고쳐라.",
                },
                {"role": "user", "content": json.dumps(parsed, ensure_ascii=False)},
            ],
        )
        try:
            parsed = _extract_json(rewrite["message"]["content"])
        except Exception:
            return _build_fallback_response(summary, evidence, question)

    if _has_foreign_script(parsed.get("explanation", "")):
        return _build_fallback_response(summary, evidence, question)

    chart_data = parsed.get("chart_data")
    if not isinstance(chart_data, list) or not chart_data:
        parsed["chart_data"] = summary.get("chart_data", [])

    insights = parsed.get("insights")
    if not isinstance(insights, list) or not insights:
        parsed["insights"] = _build_fallback_response(summary, evidence, question)["insights"]

    return parsed


def _generate_analysis_payload(df: pd.DataFrame, file_name: str, question: str) -> dict[str, Any]:
    masked = _mask_dataframe(df)
    summary = _build_summary(masked)
    evidence = _compute_stat_evidence(masked)

    # 그룹형 비교가 가능한 경우, 차트를 "열 평균"이 아닌 "그룹 평균" 기준으로 교체
    grouped_chart = _build_group_metric_chart_data(masked, evidence)
    if grouped_chart:
        summary["chart_data"] = grouped_chart["chart_data"]
        evidence["group_metric"] = {
            "group_col": grouped_chart["group_col"],
            "value_col": grouped_chart["value_col"],
            "n_groups": grouped_chart["n_groups"],
        }

    use_llm = os.environ.get("DATALENS_USE_LLM", "0") == "1"

    if use_llm:
        try:
            llm = _call_llama(question, summary, evidence)
        except Exception:
            llm = _build_fallback_response(summary, evidence, question)
    else:
        llm = _build_fallback_response(summary, evidence, question)

    recommended_method = _normalize_method_name(llm.get("recommended_method", "기술통계"))

    # 질문/데이터에 맞는 기법별 시각화 번들 생성
    viz_bundle = _build_method_visualization(
        masked,
        evidence,
        recommended_method,
        summary.get("chart_data", []),
    )

    method_options = _build_method_options(evidence)
    evidence["method_scores"] = _compute_method_scores(evidence)
    evidence["visualization"] = viz_bundle.get("chart_meta", {})
    evidence["secondary_chart_data"] = viz_bundle.get("secondary_chart_data", [])

    next_question = llm.get("next_question")
    if not isinstance(next_question, str) or not next_question.strip():
        next_question = "추가로 상관분석, 회귀분석, ANOVA, 교차분석 중 어떤 분석을 먼저 실행할까요?"

    payload = {
        "status": "success",
        "recommended_method": recommended_method,
        "explanation": llm.get("explanation", "분석 설명을 생성하지 못했습니다."),
        "insights": llm.get("insights", []),
        "chart_data": viz_bundle.get("chart_data", summary.get("chart_data", [])),
        "secondary_chart_data": viz_bundle.get("secondary_chart_data", []),
        "chart_meta": viz_bundle.get("chart_meta", {}),
        "method_options": method_options,
        "next_question": next_question,
        "table_data": summary.get("columns", []),
        "summary": {
            "row_count": summary.get("row_count", 0),
            "column_count": summary.get("column_count", 0),
            "missing_total": summary.get("missing_total", 0),
        },
        "evidence": evidence,
    }

    db = SessionLocal()
    try:
        model = AnalysisReportModel(
            file_name=file_name,
            question=question,
            recommended_method=payload["recommended_method"],
            explanation=payload["explanation"],
            insights_json=json.dumps(payload["insights"], ensure_ascii=False),
            chart_data_json=json.dumps(payload["chart_data"], ensure_ascii=False),
            table_data_json=json.dumps(payload["table_data"], ensure_ascii=False),
            summary_json=json.dumps(payload["summary"], ensure_ascii=False),
            evidence_json=json.dumps(payload["evidence"], ensure_ascii=False),
        )
        db.add(model)
        db.commit()
        db.refresh(model)
        payload["report_id"] = model.id
        payload["created_at"] = model.created_at.isoformat()
    finally:
        db.close()

    return payload


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Data Lens API Server is running. " , "status": "ok"}


@app.get("/reports")
def list_reports(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    db = SessionLocal()
    try:
        total = db.query(AnalysisReportModel).count()
        rows = (
            db.query(AnalysisReportModel)
            .order_by(AnalysisReportModel.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return {
            "total": total,
            "items": [_serialize_report(r) for r in rows],
        }
    finally:
        db.close()


@app.get("/reports/stats")
def report_stats() -> dict[str, Any]:
    db = SessionLocal()
    try:
        rows = db.query(AnalysisReportModel).all()
        total_reports = len(rows)
        total_insights = 0
        for row in rows:
            insights = _safe_json_load(row.insights_json, [])
            if isinstance(insights, list):
                total_insights += len(insights)

        recent = sorted(rows, key=lambda x: x.created_at, reverse=True)[:5]
        return {
            "total_reports": total_reports,
            "total_insights": total_insights,
            "generated_reports": total_reports,
            "recent": [
                {
                    "id": r.id,
                    "file_name": r.file_name,
                    "recommended_method": r.recommended_method,
                    "insight_count": len(_safe_json_load(r.insights_json, [])),
                    "created_at": r.created_at.isoformat(),
                }
                for r in recent
            ],
        }
    finally:
        db.close()


@app.get("/reports/{report_id}")
def get_report(report_id: int) -> dict[str, Any]:
    db = SessionLocal()
    try:
        row = db.query(AnalysisReportModel).filter(AnalysisReportModel.id == report_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="리포트를 찾을 수 없습니다.")
        return _serialize_report(row)
    finally:
        db.close()


@app.get("/reports/{report_id}/export/pdf")
def export_report_pdf(report_id: int):
    db = SessionLocal()
    try:
        row = db.query(AnalysisReportModel).filter(AnalysisReportModel.id == report_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="리포트를 찾을 수 없습니다.")

        report = _serialize_report(row)
        pdf_bytes = _build_report_pdf_bytes(report)

        stem = _safe_filename(os.path.splitext(str(row.file_name))[0], default_stem=f"report_{report_id}")
        filename = f"{stem}_report_{report_id}.pdf"
        headers = {"Content-Disposition": _content_disposition(filename)}

        return StreamingResponse(BytesIO(pdf_bytes), media_type="application/pdf", headers=headers)
    finally:
        db.close()


@app.patch("/reports/{report_id}")
def update_report(report_id: int, payload: ReportUpdateRequest) -> dict[str, Any]:
    db = SessionLocal()
    try:
        row = db.query(AnalysisReportModel).filter(AnalysisReportModel.id == report_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="리포트를 찾을 수 없습니다.")

        if payload.file_name is not None:
            new_name = payload.file_name.strip()
            if not new_name:
                raise HTTPException(status_code=400, detail="파일명은 비워둘 수 없습니다.")
            row.file_name = new_name

        db.commit()
        db.refresh(row)
        return _serialize_report(row)
    finally:
        db.close()


@app.delete("/reports/{report_id}")
def delete_report(report_id: int) -> dict[str, Any]:
    db = SessionLocal()
    try:
        row = db.query(AnalysisReportModel).filter(AnalysisReportModel.id == report_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="리포트를 찾을 수 없습니다.")
        db.delete(row)
        db.commit()
        return {"status": "success", "deleted_id": report_id}
    finally:
        db.close()


@app.post("/assistant/ask")
def assistant_ask(payload: AssistantAskRequest) -> dict[str, Any]:
    db = SessionLocal()
    try:
        row = db.query(AnalysisReportModel).filter(AnalysisReportModel.id == payload.report_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="리포트를 찾을 수 없습니다.")

        report = _serialize_report(row)
        use_llm = os.environ.get("DATALENS_USE_LLM", "0") == "1"

        if use_llm:
            try:
                prompt = {
                    "question": payload.question,
                    "report": {
                        "recommended_method": report.get("recommended_method"),
                        "explanation": report.get("explanation"),
                        "insights": report.get("insights"),
                        "summary": report.get("summary"),
                        "table_data": report.get("table_data"),
                        "chart_data": report.get("chart_data"),
                        "evidence": report.get("evidence"),
                    },
                }
                response = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {
                            "role": "system",
                            "content": "너는 데이터 분석 어시스턴트다. 반드시 제공된 report JSON만 근거로 한국어로 답하라. 숫자를 말할 때는 report의 실제 수치를 인용하고, 없는 값은 추측하지 마라. 4~6문장으로 간결하게 답하라.",
                        },
                        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                    ],
                )
                answer = response["message"]["content"]
                if _has_foreign_script(answer):
                    answer = _assistant_fallback_answer(report, payload.question)
            except Exception:
                answer = _assistant_fallback_answer(report, payload.question)
        else:
            answer = _assistant_fallback_answer(report, payload.question)

        return {
            "status": "success",
            "report_id": payload.report_id,
            "answer": answer,
        }
    finally:
        db.close()


@app.post("/analyze")
def analyze(
    file: UploadFile = File(...),
    question: str = Form("데이터의 핵심 인사이트를 설명해줘"),
) -> dict[str, Any]:
    file_name = file.filename or "uploaded_file"
    df = _read_upload(file)
    session_id = _create_dataframe_session(file_name, df)
    payload = _generate_analysis_payload(df, file_name, question)
    payload["session_id"] = session_id
    payload["is_modified"] = False
    payload.update(_build_session_grid_payload(DATAFRAME_SESSIONS[session_id]))

    return payload


@app.post("/session/analyze")
def analyze_session(payload: SessionAnalyzeRequest) -> dict[str, Any]:
    entry = _get_dataframe_session(payload.session_id)
    file_name = str(entry.get("file_name") or "uploaded_file")
    df = entry["df"]
    analysis_payload = _generate_analysis_payload(df, file_name, payload.question)
    analysis_payload["session_id"] = payload.session_id
    analysis_payload["is_modified"] = bool(entry.get("is_modified", False))
    analysis_payload.update(_build_session_grid_payload(entry))
    return analysis_payload


@app.post("/session/edit")
def edit_session(payload: SessionEditRequest) -> dict[str, Any]:
    command = (payload.command or "").strip()
    if not command:
        raise HTTPException(status_code=400, detail="수정 명령은 비워둘 수 없습니다.")

    entry = _get_dataframe_session(payload.session_id)
    current_df = entry["df"]

    try:
        code = _generate_edit_code(command, current_df)
        updated_df = _execute_edit_code(current_df, code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"데이터 수정 실패: {exc}") from exc

    entry["df"] = updated_df
    entry["is_modified"] = True
    entry["updated_at"] = _utcnow()

    file_name = str(entry.get("file_name") or "uploaded_file")
    analysis_question = (payload.question or "").strip()
    if not analysis_question:
        analysis_question = f"사용자 명령 '{command}' 적용 후 데이터 변화를 요약해줘"

    analysis_payload = _generate_analysis_payload(updated_df, file_name, analysis_question)
    analysis_payload["session_id"] = payload.session_id
    analysis_payload["is_modified"] = True
    analysis_payload["applied_code"] = code
    analysis_payload["applied_command"] = command
    analysis_payload.update(_build_session_grid_payload(entry))
    return analysis_payload


@app.post("/session/reset")
def reset_session(payload: SessionResetRequest) -> dict[str, Any]:
    entry = _get_dataframe_session(payload.session_id)

    original_df = entry.get("original_df")
    if not isinstance(original_df, pd.DataFrame):
        original_df = entry["df"].copy(deep=True)
        entry["original_df"] = original_df

    entry["df"] = original_df.copy(deep=True)
    entry["is_modified"] = False
    entry["updated_at"] = _utcnow()

    file_name = str(entry.get("file_name") or "uploaded_file")
    question = (payload.question or "").strip()
    if not question:
        question = "수정 전 원본 데이터 상태로 복원 후 다시 분석해줘"

    analysis_payload = _generate_analysis_payload(entry["df"], file_name, question)
    analysis_payload["session_id"] = payload.session_id
    analysis_payload["is_modified"] = False
    analysis_payload["reset_done"] = True
    analysis_payload.update(_build_session_grid_payload(entry))
    return analysis_payload


@app.get("/session/{session_id}/export/excel")
def export_session_excel(session_id: str):
    entry = _get_dataframe_session(session_id)
    df = entry["df"]
    excel_bytes = _build_excel_bytes(df)

    original_name = str(entry.get("file_name") or "edited_data")
    stem = _safe_filename(os.path.splitext(original_name)[0], default_stem="edited_data")
    filename = f"{stem}_edited.xlsx"
    headers = {"Content-Disposition": _content_disposition(filename)}

    media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return StreamingResponse(BytesIO(excel_bytes), media_type=media_type, headers=headers)
