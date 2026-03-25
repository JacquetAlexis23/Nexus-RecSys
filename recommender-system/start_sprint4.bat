@echo off
REM ============================================================
REM start_sprint4.bat — Arranca API + Dashboard
REM RetailRocket Recommender System — NexusDataCo
REM ============================================================
REM
REM USO:
REM   1. Abrir dos terminales en la carpeta del proyecto
REM   2. Terminal 1: python -m uvicorn src.api.main:app --reload --port 8000
REM   3. Terminal 2: streamlit run dashboard/app.py
REM
REM O ejecutar este bat para abrir ambas automaticamente:

echo.
echo ============================================================
echo  RetailRocket Recommender System - Sprint 4
echo  NexusDataCo - SoyHenry Data Science
echo ============================================================
echo.
echo Iniciando API REST en http://localhost:8000
echo Iniciando Dashboard en http://localhost:8501
echo.
echo Presiona Ctrl+C en cada ventana para detener.
echo.

REM Abrir API en nueva ventana
start "API RetailRocket" cmd /k "python -m uvicorn src.api.main:app --reload --port 8000"

REM Esperar 3 segundos para que la API cargue los modelos
timeout /t 5 /nobreak > nul

REM Abrir Dashboard en nueva ventana
start "Dashboard RetailRocket" cmd /k "streamlit run dashboard/app.py"

echo.
echo Ambos servicios iniciados.
echo API:       http://localhost:8000
echo Dashboard: http://localhost:8501
echo Docs API:  http://localhost:8000/docs
echo.
pause
