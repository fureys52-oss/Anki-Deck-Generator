@echo off
echo Starting the Anki Deck Generator...
echo If a firewall permission dialog appears, please grant access.

:: This command ensures the script runs in the correct directory
cd /d "%~dp0"

:: Activate the virtual environment before running the Python script
echo Activating virtual environment...
CALL venv\scripts\activate.bat

:: Now, run the application using the venv's Python
echo Launching the application...
python app.py

pause