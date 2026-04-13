@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

set "PROJECT_ROOT=%~dp0"
set "PYTHON_EXE=%PROJECT_ROOT%venv\Scripts\python.exe"
set "MAIN_FILE=%PROJECT_ROOT%main_ai.py"
set "RECREATE_VENV=0"

if exist "%PYTHON_EXE%" (
  "%PYTHON_EXE%" -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo [안내] 기존 venv가 손상되어 재생성합니다.
    set "RECREATE_VENV=1"
  )
) else (
  set "RECREATE_VENV=1"
)

if "%RECREATE_VENV%"=="1" (
  set "BOOTSTRAP_PY="
  if exist "%LocalAppData%\Microsoft\WindowsApps\python3.13.exe" (
    set "BOOTSTRAP_PY=%LocalAppData%\Microsoft\WindowsApps\python3.13.exe"
  )
  if not defined BOOTSTRAP_PY (
    where python >nul 2>nul
    if not errorlevel 1 set "BOOTSTRAP_PY=python"
  )
  if not defined BOOTSTRAP_PY (
    where py >nul 2>nul
    if not errorlevel 1 set "BOOTSTRAP_PY=py -3"
  )

  if not defined BOOTSTRAP_PY (
    echo [오류] venv를 재생성할 Python 실행 파일을 찾지 못했습니다.
    echo Python 설치 후 다시 실행해 주세요.
    exit /b 1
  )

  if exist "%PROJECT_ROOT%venv" rmdir /s /q "%PROJECT_ROOT%venv"
  echo [안내] 새 가상환경을 생성합니다: %PROJECT_ROOT%venv
  !BOOTSTRAP_PY! -m venv "%PROJECT_ROOT%venv"
  if errorlevel 1 (
    echo [오류] 가상환경 생성에 실패했습니다.
    exit /b 1
  )
)

if not exist "%PYTHON_EXE%" (
  echo [오류] 가상환경 Python을 찾지 못했습니다: %PYTHON_EXE%
  echo 먼저 프로젝트 루트에서 python -m venv venv 를 실행해 venv를 만들어 주세요.
  exit /b 1
)

if not exist "%MAIN_FILE%" (
  echo [오류] 실행 파일을 찾지 못했습니다: %MAIN_FILE%
  exit /b 1
)

"%PYTHON_EXE%" "%MAIN_FILE%"
if errorlevel 1 (
  echo [실패] main_ai.py 실행 중 오류가 발생했습니다. 위 메시지를 확인해 주세요.
  exit /b 1
)
