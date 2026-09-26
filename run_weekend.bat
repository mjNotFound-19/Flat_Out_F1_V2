@echo off
REM Flat Out F1 v3 - full loop: sync -> build -> score finished races -> retrain -> predict next race.
REM Safe to run after every session; pass extra args through, e.g.  run_weekend.bat --sims 1000000
cd /d "%~dp0"
python -m flatout weekend %*
