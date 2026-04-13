import ollama
import re
import json
from pathlib import Path

# 1. 인공지능에게 규칙 가르치기 (이게 바로 튜닝입니다!)
# 재윤님이 만든 '스키마'와 '결과값'을 어떻게 해석할지 알려주는 지시서입니다.
SYSTEM_PROMPT = """
너는 데이터 분석 팀 '데이터 렌즈'의 AI 전문가야.
사용자가 데이터를 올리면, 너는 재윤이가 분석한 통계 수치를 바탕으로 설명해줘야 해.

[규칙]
1. 분석 결과의 p-value가 0.05보다 작으면 "통계적으로 유의미한 차이가 있다"고 설명해.
2. 전문 용어보다는 일반인이 이해하기 쉽게 비유를 들어줘.
3. 보안을 위해 절대 사람 이름이나 전화번호는 언급하지 마.
4. 반드시 한국어만 사용하고, 영어/일본어/중국어를 섞지 마.
5. 답변은 4~7문장으로 간결하게 작성해.
6. 숫자 해석은 결과에서 제공된 값만 사용하고 임의 추측은 하지 마.
"""


def ask_datalens_ai(user_question, data_info):
    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f"데이터 정보: {data_info}\n사용자 질문: {user_question}"}
    ])
    return response['message']['content']


def rewrite_korean_only(text):
    """모델 응답을 한국어만 사용한 자연스러운 문장으로 교정한다."""
    response = ollama.chat(model='llama3.2', messages=[
        {
            'role': 'system',
            'content': (
                "너는 한국어 교정기다. 입력 문장의 의미를 유지하면서 한국어만 사용해 자연스럽게 고쳐라. "
                "영어/일본어/중국어를 절대 사용하지 마라. "
                "통계 용어는 한국어로 바꿔라(예: p-value -> 유의확률). "
                "4~7문장으로 간결하게 작성하라."
            )
        },
        {'role': 'user', 'content': text}
    ])
    return response['message']['content']


def has_foreign_script(text):
    """영문/일문/중문이 포함되어 있는지 확인한다."""
    return bool(re.search(r"[A-Za-z\u3040-\u30FF\u4E00-\u9FFF]", text))


def clean_to_korean_charset(text):
    """한글/숫자/기본 문장부호만 남기고 정리한다."""
    cleaned = re.sub(r"[^0-9가-힣ㄱ-ㅎㅏ-ㅣ\s\.,!?;:%()\-\"'\n]", "", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def enforce_korean_only(text):
    """외국어가 섞이면 최대 2회 재작성 후 문자셋 정리를 수행한다."""
    rewritten = text
    for _ in range(2):
        if not has_foreign_script(rewritten):
            break
        rewritten = rewrite_korean_only(rewritten)
    return clean_to_korean_charset(rewritten)


def _find_latest_file(directory: Path, pattern: str):
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_project_data_info():
    """project_dataAnalyzing의 JSON 결과를 읽어 프롬프트 문자열로 구성한다."""
    project_root = Path(__file__).resolve().parents[1]
    analyzing_dir = project_root / "project_dataAnalyzing"

    if not analyzing_dir.exists():
        raise FileNotFoundError(f"분석 프로젝트 폴더를 찾을 수 없습니다: {analyzing_dir}")

    metadata_file = _find_latest_file(analyzing_dir, "*_metadata.json")
    analysis_file = _find_latest_file(analyzing_dir, "*_analysis.json")

    if metadata_file is None or analysis_file is None:
        missing = []
        if metadata_file is None:
            missing.append("*_metadata.json")
        if analysis_file is None:
            missing.append("*_analysis.json")
        raise FileNotFoundError(
            "분석 JSON 파일이 없습니다. 먼저 project_dataAnalyzing에서 생성하세요: "
            + ", ".join(missing)
        )

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_file.read_text(encoding="utf-8"))

    return (
        "[메타데이터 JSON]\n"
        + json.dumps(metadata, ensure_ascii=False, indent=2)
        + "\n\n[분석 계획 JSON]\n"
        + json.dumps(analysis, ensure_ascii=False, indent=2)
    )


# 2. 실제로 재윤님이 준 샘플 데이터(메타데이터+통계결과) 넣기
sample_schema = """
[메타데이터]
데이터 크기: 100행 × 10개 변수

변수 목록:
- 나이: 연속형 | 범위 24.0~55.0 | 평균 39.73 | 정규성=없음
- 성별: 이진형 | 값: ['여', '남']
- 직군: 범주형 | 수준 4개: ['마케팅', '개발', '디자인', '기획']
- 주간근무시간: 연속형 | 범위 35.0~72.0 | 평균 53.69 | 정규성=없음
- 연봉_만원: 연속형 | 범위 4447.0~7662.0 | 평균 6003.27 | 정규성=있음
- 수면시간: 연속형 | 범위 5.0~9.5 | 평균 6.915 | 정규성=있음
- 스트레스점수: 연속형 | 범위 10.0~77.0 | 평균 41.51 | 정규성=있음
- 번아웃여부: 이진형 | 값: [0, 1]
- 운동빈도: 범주형 | 수준 3개: ['주1-2회', '안함', '주3회이상']
- 직무만족도: 연속형 | 범위 1.0~5.0 | 평균 3.315 | 정규성=있음

[핵심 통계 결과]
1) t_test (성별 vs 연봉_만원)
- p-value: 0.0013
- 해석: 남(M=5770.77)와 여(M=6217.88)의 연봉 차이는 통계적으로 유의미함.

2) anova (직군 vs 스트레스점수)
- p-value: 0.173
- 해석: 직군별 스트레스 차이는 통계적으로 유의미하지 않음.

3) correlation (나이, 연봉_만원, 스트레스점수)
- 유의 상관쌍: 나이-연봉_만원 (r=0.766, p=0.0)

4) regression (연봉_만원 예측)
- R²: 0.6874, adj.R²: 0.6776, 모형 p-value: 0.0
- 유의한 변수: 나이, 주간근무시간, 스트레스점수
"""

def main():
    default_q = "남녀 연봉 차이가 정말 있는 거야?"
    user_q = input(f"질문을 입력하세요 (엔터 시 기본 질문 사용)\n> ").strip()
    if not user_q:
        user_q = default_q

    print("인공지능이 생각 중입니다...")
    try:
        try:
            data_info = load_project_data_info()
        except Exception as data_err:
            print("[안내] project_dataAnalyzing 결과 파일을 찾지 못해 샘플 데이터로 실행합니다.")
            print(f"- 상세: {data_err}")
            data_info = sample_schema

        result = ask_datalens_ai(user_q, data_info)
        result = enforce_korean_only(result)
        print("\n[AI 답변]:", result)
    except Exception as e:
        print("\n[오류] AI 호출에 실패했습니다.")
        print("- Ollama 앱/서버가 실행 중인지 확인")
        print("- llama3.2 모델이 설치되어 있는지 확인 (예: ollama pull llama3.2)")
        print(f"- 상세 오류: {e}")


if __name__ == "__main__":
    main()
