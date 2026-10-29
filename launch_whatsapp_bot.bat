@echo off
title AI Fashion Assistant - WhatsApp Webhook Server
echo =======================================================
echo Starting WhatsApp Flask Webhook Server on Port 8080...
echo =======================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python src\app.py
pause
