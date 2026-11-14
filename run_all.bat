@echo off
REM HeatSense: run the whole pipeline in order (Windows).
setlocal
cd /d "%~dp0"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
echo [run_all] using %PY%

REM Only regenerate synthetic data if the grid is missing or synthetic.
set "REGEN=1"
if exist "data\grid_features.csv" (
  for /f "usebackq delims=" %%s in (`%PY% -c "import pandas as pd;print(pd.read_csv('data/grid_features.csv',usecols=['data_source'],nrows=1).data_source.iloc[0])" 2^>nul`) do (
    if not "%%s"=="synthetic" (
      set "REGEN=0"
      echo [run_all] keeping existing grid_features.csv ^(data_source=%%s^)
    )
  )
)

if "%REGEN%"=="1" (
  %PY% -m pipeline.synthetic || goto :error
)
%PY% -m pipeline.features || goto :error
%PY% -m pipeline.model || goto :error
%PY% -m pipeline.cooling || goto :error
%PY% -m pipeline.priority || goto :error
%PY% -m pipeline.summaries || goto :error
%PY% -m pipeline.outputs || goto :error

echo [run_all] done. Start the app with: %PY% app\app.py
exit /b 0

:error
echo [run_all] FAILED with exit code %errorlevel%
exit /b %errorlevel%
