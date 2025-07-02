@echo off
setlocal enabledelayedexpansion

echo ====================================
echo Python Project Auto-Setup (Windows)
echo ====================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python or Miniconda first.
    echo.
    echo Option 1: Install Miniconda manually from: https://www.anaconda.com/download/
    echo Option 2: Install Python from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo Python checked

REM Try to find conda installation
set "CONDA_PATH="
for %%i in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\Anaconda3" "%ProgramData%\Miniconda3") do (
    if exist "%%i\Scripts\conda.exe" (
        set "CONDA_PATH=%%i"
		echo Found Conda path, adding to temp PATH variable for the installation
        goto :found_conda
		
    )
)
echo Conda not found on your machine, please install it it and try again.
echo Link: https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation
pause

:found_conda
echo Conda is available at %CONDA_PATH%
set "PATH=%CONDA_PATH%;%CONDA_PATH%\Scripts;%CONDA_PATH%\Library\bin;%PATH%"
REM Check if main.py exists
if not exist "main.py" (
    echo Error: main.py not found in current directory!
    echo Please make sure main.py is in the same folder as this script.
    pause
    exit /b 1
)

REM Check for dependency files in order of preference
set DEPS_FILE=
set DEPS_TYPE=
if exist "environment.yml" (
    set DEPS_FILE=environment.yml
    set DEPS_TYPE=conda
    echo Found environment.yml - using Conda
) else (
    echo Warning: No dependency file found!
    pause
    exit /b 1
)

REM Set environment name
set ENV_NAME=discord_bot

echo Step 1: Setting up virtual environment...
echo.

if !CONDA_AVAILABLE! equ 0 (
    echo Using Conda for environment management...
    
    REM Check if conda environment already exists
    conda env list | findstr "!ENV_NAME!" >nul 2>&1
    if not errorlevel 1 (
        echo Conda environment '!ENV_NAME!' already exists.
    ) else (
        echo Creating new conda environment '!ENV_NAME!'...
        conda env create -f environment.yml
        if errorlevel 1 (
            echo Failed to create conda environment!
            pause
            exit /b 1
        )
    )
    
    
)

echo.
echo Step 2: Setting up environment variables...
echo.

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating .env file...
    (
        echo # API Keys - Replace with your actual API keys
        echo DISCORD_TOKEN="YOUR_DISCORD_TOKEN_HERE"
        echo CLAUDE_API_KEY="YOUR_CLAUDE_API_KEY_HERE"
        echo GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
        echo OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
        echo CUSTOM_API_KEY="YOUR_CUSTOM_API_KEY_HERE"
    ) > .env
    echo.
    echo .env file created! Please edit .env and add your actual API keys.
    echo WARNING: Never commit .env files to version control!
    echo.
	notepad .env
) else (
    echo .env file already exists.
)

REM Create .gitignore if it doesn't exist to protect .env file
if not exist ".gitignore" (
    echo Creating .gitignore to protect sensitive files...
    (
        echo # Environment variables
        echo .env
        echo.
        echo # Bot Data
        echo bot_data/
    ) > .gitignore
    echo .gitignore created to protect your .env file!
)

echo.
echo Step 3: Installacion complete. Before running the application, please set up your API KEYS in .env file
echo.
echo Press any key to exit...
pause >nul