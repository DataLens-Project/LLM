# PDF 한글 폰트 배포 체크리스트

## 목표
- PDF 다운로드 시 한글이 깨지지 않도록 서버에서 유니코드 폰트를 반드시 로드한다.

## 1) 폰트 파일 배치
- 서버(EC2/컨테이너)에 아래 중 하나를 배치한다.
  - NanumGothic.ttf
  - NotoSansKR-Regular.ttf
- 권장 경로:
  - /app/fonts/NanumGothic.ttf
  - /app/fonts/NotoSansKR-Regular.ttf

## 2) 환경변수 설정
- 애플리케이션 실행 환경에 아래 변수 설정:
  - DATALENS_PDF_FONT_PATH=/app/fonts/NanumGothic.ttf

### EC2/Linux 즉시 점검 커맨드
1. 폰트 폴더 생성 및 파일 복사
  - mkdir -p /app/fonts
  - cp ./NanumGothic.ttf /app/fonts/NanumGothic.ttf
2. 파일 확인
  - ls -l /app/fonts
3. 환경변수 설정(현재 셸)
  - export DATALENS_PDF_FONT_PATH=/app/fonts/NanumGothic.ttf
4. 값 확인
  - echo $DATALENS_PDF_FONT_PATH

### Docker/Compose 점검 커맨드
1. 컨테이너 내부 폰트 파일 확인
  - docker exec -it <container_name> ls -l /app/fonts
2. 컨테이너 환경변수 확인
  - docker exec -it <container_name> printenv DATALENS_PDF_FONT_PATH

## 3) 서버 재시작
- 환경변수 적용 후 API 서버를 재시작한다.

### 재시작 예시
- systemd:
  - sudo systemctl restart datalens-api
  - sudo systemctl status datalens-api --no-pager
- docker compose:
  - docker compose up -d --force-recreate
  - docker compose ps

## 4) 동작 검증
1. CSV 업로드 후 분석 수행
2. PDF 다운로드 실행
3. PDF에서 한글 문장 확인
   - 추천 기법
   - 설명
   - 인사이트
4. 서버 폰트 상태 API 확인
  - curl -s http://<API_HOST>:8000/debug/pdf-font-status
  - 기대값:
    - full_unicode: true
    - using_fallback_helvetica: false
    - font_name: Helvetica가 아님

## 5) 장애 시 점검
- 폰트 파일 존재 여부 확인
- 컨테이너 내부 경로와 환경변수 경로 일치 여부 확인
- 읽기 권한 확인
- 애플리케이션 로그에서 폰트 로딩 실패 여부 확인
- /debug/pdf-font-status 결과가 Helvetica면 폰트 로딩 실패로 판단

## 참고
- 코드 내 폰트 탐색 순서에 기본 경로 후보가 포함되어 있어도, 운영에서는 DATALENS_PDF_FONT_PATH를 명시적으로 지정하는 방식이 가장 안정적이다.
