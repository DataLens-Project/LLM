$ErrorActionPreference = 'Stop'

# Ensure Korean output is not garbled in legacy console hosts.
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

# Always run from project root regardless of current terminal location.
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
$mainFile = Join-Path $projectRoot "main_ai.py"

function Get-VenvBaseExecutable {
    param([string]$VenvConfigPath)

    if (-not (Test-Path $VenvConfigPath)) {
        return $null
    }

    $line = Get-Content $VenvConfigPath | Where-Object { $_ -like 'executable = *' } | Select-Object -First 1
    if (-not $line) {
        return $null
    }

    return ($line -replace '^executable\s*=\s*', '').Trim()
}

function Ensure-Venv {
    param(
        [string]$ProjectRoot,
        [string]$PythonExePath
    )

    $venvDir = Join-Path $ProjectRoot "venv"
    $venvCfg = Join-Path $venvDir "pyvenv.cfg"
    $recreate = $false

    if (Test-Path $PythonExePath) {
        $baseExe = Get-VenvBaseExecutable -VenvConfigPath $venvCfg
        if ($baseExe -and -not (Test-Path $baseExe)) {
            Write-Host "[안내] 기존 venv의 기준 Python 경로가 사라져 venv를 재생성합니다." -ForegroundColor Yellow
            Write-Host "       누락된 경로: $baseExe" -ForegroundColor Yellow
            $recreate = $true
        }
    } else {
        $recreate = $true
    }

    if (-not $recreate) {
        return
    }

    $bootstrapPython = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $bootstrapPython) {
        Write-Host "[오류] 시스템에서 python 명령을 찾지 못했습니다." -ForegroundColor Red
        Write-Host "먼저 Python을 설치하고, 'python --version'이 동작하는지 확인해 주세요."
        exit 1
    }

    if (Test-Path $venvDir) {
        Remove-Item -Path $venvDir -Recurse -Force
    }

    Write-Host "[안내] 새 가상환경을 생성합니다: $venvDir"
    & $bootstrapPython -m venv $venvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $PythonExePath)) {
        Write-Host "[오류] venv 생성에 실패했습니다." -ForegroundColor Red
        Write-Host "다음 명령으로 수동 확인해 주세요: python -m venv venv"
        exit 1
    }
}

Ensure-Venv -ProjectRoot $projectRoot -PythonExePath $pythonExe

if (-not (Test-Path $pythonExe)) {
    Write-Host "[오류] 가상환경 Python을 찾지 못했습니다: $pythonExe" -ForegroundColor Red
    Write-Host "먼저 프로젝트 루트에서 python -m venv venv 를 실행해 venv를 만들어 주세요."
    exit 1
}

if (-not (Test-Path $mainFile)) {
    Write-Host "[오류] 실행 파일을 찾지 못했습니다: $mainFile" -ForegroundColor Red
    exit 1
}

& $pythonExe $mainFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "[실패] main_ai.py 실행 중 오류가 발생했습니다. 위 메시지를 확인해 주세요." -ForegroundColor Red
    exit $LASTEXITCODE
}
