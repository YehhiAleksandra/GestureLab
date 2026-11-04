@echo off
chcp 65001 >nul
echo Запуск Telegram-бота с капчей...
cd /d "%~dp0"
python bot.py
pause
