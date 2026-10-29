@echo off
title AI Fashion Assistant - Web Demo
echo =======================================================
echo Starting AI Fashion Assistant Web Demo (Streamlit)...
echo =======================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m streamlit run demo_web.py
pause
