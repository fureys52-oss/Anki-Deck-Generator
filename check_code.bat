@echo off
REM ==============================================================================
REM Anki Deck Generator - Code Quality & Bug Check Script
REM
REM This script provides two modes for checking the Python code quality:
REM  1. Critical Bug Check: Fast check for syntax errors and undefined variables.
REM  2. Full Style Check:   Comprehensive check for all bugs and style guide violations.
REM ==============================================================================

setlocal

REM Ensure the script runs relative to its own location
cd /d "%~dp0"

REM --- SCRIPT SETUP ---
echo Activating virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment 'venv' not found in this directory.
    echo Please create it and install flake8 first.
    pause
    exit /b 1
)
call venv\Scripts\activate

REM Create the reports directory if it doesn't exist
if not exist "quality_reports" mkdir "quality_reports"

REM Generate a timestamp for the log file
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set "TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%"

REM --- USER MENU ---
:menu
cls
echo =======================================================
echo         Anki Deck Generator - CODE ANALYSIS
echo =======================================================
echo.
echo Please choose an analysis type:
echo.
echo   1) Critical Bug Check (Fast)
echo      - Checks for syntax errors, undefined names, etc. (Codes: F, E9)
echo.
echo   2) Full Style & Bug Check (Comprehensive)
echo      - Checks all Flake8 rules for bugs and code style.
echo.
set /p "choice=Enter your choice (1 or 2): "

if "%choice%"=="1" goto :critical_check
if "%choice%"=="2" goto :full_check

echo Invalid choice. Please enter 1 or 2.
timeout /t 2 >nul
goto :menu


REM --- ANALYSIS CONFIGURATION ---
:critical_check
set "CHECK_TYPE=Critical Errors Only"
set "LOG_FILE=quality_reports\critical_errors_%TIMESTAMP%.txt"
set "FLAKE8_ARGS=--select=F,E9"
goto :run_analysis

:full_check
set "CHECK_TYPE=Full Style & Bug Check"
set "LOG_FILE=quality_reports\full_style_report_%TIMESTAMP%.txt"
REM NEW: Added --max-line-length for more modern code formatting. Default is a very strict 79.
set "FLAKE8_ARGS=--max-line-length=99"
goto :run_analysis


REM --- EXECUTION ---
:run_analysis
echo.
echo Running flake8 analysis (%CHECK_TYPE%)...
echo Saving report to %LOG_FILE%

REM NEW: Added .pdf_cache, .ai_cache, and logs to the exclude list.
flake8 . %FLAKE8_ARGS% --exclude=venv,.git,__pycache__,.vscode,.vs,.pdf_cache,.ai_cache,logs > %LOG_FILE%


REM --- REPORTING ---
echo.
echo Analysis complete.

REM Check if the report file contains any content.
findstr . "%LOG_FILE%" >nul
if %errorlevel% equ 0 (
    echo.
    echo =======================================================
    echo  RESULT: WARNING! Issues found.
    echo  Opening report: %LOG_FILE%
    echo =======================================================
    start "" "%LOG_FILE%"
) else (
    echo.
    echo =======================================================
    echo  RESULT: SUCCESS! No issues found for this check.
    echo =======================================================
    del "%LOG_FILE%"
)

echo.
endlocal
pause