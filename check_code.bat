@echo off
REM This is the "Targeted" version of the script.
REM It uses --exclude to ignore irrelevant folders like the virtual environment.
echo =======================================================
echo  Anki Deck Generator - TARGETED BUG CHECK
echo =======================================================

REM Ensure the script runs relative to its own location
cd /d "%~dp0"

REM Create the reports directory if it doesn't exist
if not exist "quality_reports" mkdir "quality_reports"

REM Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Generate a timestamp for the log file
echo Generating timestamp...
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%
set LOG_FILE=quality_reports\critical_errors_%TIMESTAMP%.txt

REM =================================================================
REM  THE KEY CHANGE IS THE --exclude FLAG
REM  This tells flake8 to completely ignore these folders.
REM =================================================================
echo.
echo Running targeted flake8 analysis (Critical Errors Only)...
echo Saving report to %LOG_FILE%
flake8 . --select=F,E9 --exclude=venv,.git,__pycache__,.vscode,.vs > %LOG_FILE%

echo.
echo Analysis complete.

REM Check if the report file contains any content.
findstr . "%LOG_FILE%" >nul
if %errorlevel% equ 0 (
    echo.
    echo =======================================================
    echo  RESULT: WARNING! Critical bugs found.
    echo  Opening report: %LOG_FILE%
    echo =======================================================
    start "" "%LOG_FILE%"
) else (
    echo.
    echo =======================================================
    echo  RESULT: SUCCESS! No critical bugs found in your code.
    echo =======================================================
    del "%LOG_FILE%"
)

echo.
pause