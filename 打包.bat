@echo off
setlocal EnableDelayedExpansion

title ChickenRice v2 Release Packager

echo ========================================
echo Preparing Release Environment...
echo ========================================

rem Set 7z path
set "SEVEN_ZIP=C:\Program Files\7-Zip-Zstandard\7z.exe"

if not exist "%SEVEN_ZIP%" (
    echo [ERROR] 7-Zip not found at "%SEVEN_ZIP%"
    echo Please install 7-Zip Zstandard or update the path in this script.
    pause
    exit /b 1
)

echo Using 7-Zip: "%SEVEN_ZIP%"

rem Activate Virtual Environment and Build EXE
echo.
echo ========================================
echo Building Executable...
echo ========================================

rem Assume venv is at .\venv
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" build_exe.py
) else (
    echo [ERROR] Virtual environment not found at .\venv
    pause
    exit /b 1
)

if errorlevel 1 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Packaging Files...
echo ========================================

set "DIST_DIR=dist_release\ChickenRice_v2"
set "RELEASE_DIR=releases"

if not exist "%RELEASE_DIR%" mkdir "%RELEASE_DIR%"

rem Copy documentation and config
copy "README.md" "%DIST_DIR%\"
copy "使用说明.txt" "%DIST_DIR%\"
copy "generation_config.json5" "%DIST_DIR%\"
copy "requirements.txt" "%DIST_DIR%\"
copy "LICENSE" "%DIST_DIR%\"

rem Create models directory structure in dist
if not exist "%DIST_DIR%\models" mkdir "%DIST_DIR%\models"

rem Copy VAD models (Always needed)
copy "models\whisper_vad.onnx" "%DIST_DIR%\models\"
copy "models\whisper_vad_metadata.json" "%DIST_DIR%\models\"

rem Copy VAD config files for offline execution
if not exist "%DIST_DIR%\models\whisper-base" mkdir "%DIST_DIR%\models\whisper-base"
copy "models\whisper-base\*" "%DIST_DIR%\models\whisper-base\"


echo.
echo ========================================
echo All Tasks Completed!
echo Releases are in: %RELEASE_DIR%
echo ========================================
pause
