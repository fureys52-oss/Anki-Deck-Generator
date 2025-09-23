@echo off
REM This is the "Targeted" version of the script.
REM It uses --exclude to ignore irrelevant folders like the virtual environment.
echo =======================================================
echo  Anki Deck Generator - TARGETED BUG CHECK
echo =======================================================

REM Create the reports directory if it doesn't exist
if not exist "quality_reports" mkdir "quality_reports"

REM Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Generate a timestamp for the log file
echo Generating timestamp...
set TIMESTAMP=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
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

REM Check if the report file is empty or not
for %%A in (%LOG_FILE%) do set FileSize=%%~zA
if %FileSize% equ 0 (
    echo.
    echo =======================================================
    echo  RESULT: SUCCESS! No critical bugs found in your code.
    echo =======================================================
    del %LOG_FILE%
) else (
    echo.
    echo =======================================================
    echo  RESULT: WARNING! Critical bugs found.
    echo  Opening report: %LOG_FILE%
    echo =======================================================
    start "" %LOG_FILE%
)

echo.
pause