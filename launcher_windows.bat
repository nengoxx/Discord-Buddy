@echo off

REM Try to find conda installation
set "CONDA_PATH="
for %%i in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\Anaconda3" "%ProgramData%\Miniconda3") do (
    if exist "%%i\Scripts\conda.exe" (
        set "CONDA_PATH=%%i"
        echo Found Conda at %%i
        goto :found_conda
    )
)

echo Conda not found on your machine, please install it and try again.
pause
exit /b 1

:found_conda
echo Conda is available at %CONDA_PATH%
set "PATH=%CONDA_PATH%;%CONDA_PATH%\Scripts;%CONDA_PATH%\Library\bin;%PATH%"


REM Check if discord_bot environment exists
set "ENV_PATH=%CONDA_PATH%\envs\discord_bot"
if not exist "%ENV_PATH%" (
    echo discord_bot environment not found at %ENV_PATH%
    echo Please create the environment first: conda create -n discord_bot python
    pause
    exit /b 1
)

REM Activate discord_bot environment by setting environment variables
echo Activating discord_bot environment...
set "CONDA_DEFAULT_ENV=discord_bot"
set "CONDA_PREFIX=%ENV_PATH%"
set "PATH=%ENV_PATH%;%ENV_PATH%\Scripts;%ENV_PATH%\Library\bin;%PATH%"

echo Environment activated. Python path:
where python

echo Running main.py...
python main.py

pause