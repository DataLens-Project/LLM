import json
import os
import re
from io import BytesIO
from typing import Any
from datetime import datetime

import ollama
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
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


def _read_upload(file: UploadFile) -> pd.DataFrame:
    filename = (file.filename or "").lower()
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    try:
        if filename.endswith(".csv"):
            return pd.read_csv(BytesIO(raw))
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            return pd.read_excel(BytesIO(raw))
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

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
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

    return {
        "row_count": int(rows),
        "column_count": int(cols),
        "missing_total": missing,
        "columns": column_summaries,
        "chart_data": chart_data,
    }


def _compute_stat_evidence(df: pd.DataFrame) -> dict[str, Any]:
    evidence: dict[str, Any] = {"t_test": None, "correlation": None}

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    # 1) 이진 집단 + 수치형 1개에 대한 t-test
    for gcol in categorical_cols:
        non_null = df[gcol].dropna()
        if non_null.nunique() != 2:
            continue
        levels = list(non_null.unique())
        for vcol in numeric_cols:
            subset = df[[gcol, vcol]].dropna()
            if len(subset) < 10:
                continue
            g1 = subset[subset[gcol] == levels[0]][vcol]
            g2 = subset[subset[gcol] == levels[1]][vcol]
            if len(g1) < 3 or len(g2) < 3:
                continue

            t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
            evidence["t_test"] = {
                "group_col": str(gcol),
                "value_col": str(vcol),
                "group_a": str(levels[0]),
                "group_b": str(levels[1]),
                "mean_a": round(float(g1.mean()), 3),
                "mean_b": round(float(g2.mean()), 3),
                "t_stat": round(float(t_stat), 3),
                "p_value": round(float(p_val), 4),
            }
            break
        if evidence["t_test"]:
            break

    # 2) 수치형 간 상관분석에서 절대값 기준 가장 강한 쌍
    if len(numeric_cols) >= 2:
        best_pair = None
        best_abs_r = -1.0
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                subset = df[[c1, c2]].dropna()
                if len(subset) < 5:
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

    return evidence


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
        "created_at": model.created_at.isoformat(),
    }


def _has_foreign_script(text: str) -> bool:
    return bool(re.search(r"[A-Za-z\u3040-\u30FF\u4E00-\u9FFF]", text))


def _build_fallback_response(summary: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    t_e = evidence.get("t_test")
    c_e = evidence.get("correlation")

    if t_e:
        sig = "통계적으로 유의미" if t_e["p_value"] < 0.05 else "통계적으로 유의미하지 않음"
        recommended = "T-검정"
        explain = (
            f"{t_e['group_col']}에 따른 {t_e['value_col']} 차이를 확인하기 위해 T-검정을 적용했습니다. "
            f"{t_e['group_a']} 평균은 {t_e['mean_a']}, {t_e['group_b']} 평균은 {t_e['mean_b']}입니다. "
            f"검정 결과 p값은 {t_e['p_value']}로 {sig}으로 해석됩니다. "
            "해석 시 표본 수와 결측치 규모를 함께 점검하는 것이 좋습니다."
        )
    elif c_e:
        recommended = "상관분석"
        explain = (
            f"{c_e['col_x']}와 {c_e['col_y']}의 상관분석을 수행했습니다. "
            f"상관계수 r은 {c_e['r']}이고 p값은 {c_e['p_value']}입니다. "
            "상관의 크기와 방향을 기반으로 변수 간 관계를 해석할 수 있습니다. "
            "다만 상관관계는 인과관계를 의미하지 않으므로 추가 검증이 필요합니다."
        )
    else:
        recommended = "기술통계"
        explain = "현재 데이터로는 기본 기술통계를 중심으로 분포, 결측치, 변수 구성을 먼저 점검하는 것이 적절합니다."

    insights = [
        f"데이터는 총 {summary.get('row_count', 0)}행, {summary.get('column_count', 0)}열입니다.",
        f"전체 결측치는 {summary.get('missing_total', 0)}개입니다.",
        "의사결정 전에 유의확률과 효과크기를 함께 확인하세요.",
    ]

    return {
        "recommended_method": recommended,
        "explanation": explain,
        "insights": insights,
        "chart_data": summary.get("chart_data", []),
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
        return _build_fallback_response(summary, evidence)

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
            return _build_fallback_response(summary, evidence)

    if _has_foreign_script(parsed.get("explanation", "")):
        return _build_fallback_response(summary, evidence)

    chart_data = parsed.get("chart_data")
    if not isinstance(chart_data, list) or not chart_data:
        parsed["chart_data"] = summary.get("chart_data", [])

    insights = parsed.get("insights")
    if not isinstance(insights, list) or not insights:
        parsed["insights"] = _build_fallback_response(summary, evidence)["insights"]

    return parsed


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


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


@app.post("/analyze")
def analyze(
    file: UploadFile = File(...),
    question: str = Form("데이터의 핵심 인사이트를 설명해줘"),
) -> dict[str, Any]:
    file_name = file.filename or "uploaded_file"
    df = _read_upload(file)
    masked = _mask_dataframe(df)
    summary = _build_summary(masked)
    evidence = _compute_stat_evidence(masked)

    use_llm = os.environ.get("DATALENS_USE_LLM", "0") == "1"

    if use_llm:
        try:
            llm = _call_llama(question, summary, evidence)
        except Exception:
            llm = _build_fallback_response(summary, evidence)
    else:
        llm = _build_fallback_response(summary, evidence)

    payload = {
        "status": "success",
        "recommended_method": llm.get("recommended_method", "기술통계"),
        "explanation": llm.get("explanation", "분석 설명을 생성하지 못했습니다."),
        "insights": llm.get("insights", []),
        "chart_data": llm.get("chart_data", summary.get("chart_data", [])),
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
