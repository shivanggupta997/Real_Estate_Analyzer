@echo off

REM --- Start Frontend in New Window ---
start "Frontend" cmd /k "cd Chat_bot && npm install && npm run dev"

REM --- Backend Setup ---
cd backend

REM Optional: activate venv if you're using one
:: call venv\Scripts\activate

REM Install required Python packages
pip install -r requirements.txt

REM Run Django setup
call python manage.py migrate
call python manage.py runserver
