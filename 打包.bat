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

rem --------------------------------------------------
rem 1. Create No-Model Package (ZIP)
rem --------------------------------------------------
echo.
echo Creating No-Model Package (ZIP)...
set "ZIP_NAME=%RELEASE_DIR%\ChickenRice_v2_NoModel.zip"
if exist "%ZIP_NAME%" del "%ZIP_NAME%"

rem Use 7z to create zip for better compatibility and speed
rem Zip the folder 'ChickenRice_v2' so it extracts as a folder
"%SEVEN_ZIP%" a -tzip "%ZIP_NAME%" "dist_release\ChickenRice_v2"

echo Created: %ZIP_NAME%

rem --------------------------------------------------
rem 2. Create Full Package (7z Split)
rem --------------------------------------------------
echo.
echo Creating Full Package (7z Split)...

rem Copy the large model to dist for packaging
echo Copying Large Model (this may take a while)...
set "MODEL_SRC=%CD%\models\whisper-chickenrice-large-v2-ov"
set "MODEL_DST=%CD%\%DIST_DIR%\models\whisper-chickenrice-large-v2-ov"

echo Model Source: "%MODEL_SRC%"
echo Model Dest: "%MODEL_DST%"

if not exist "%MODEL_SRC%" (
    echo [WARNING] Large model not found at "%MODEL_SRC%"
    echo Skipping Full Package creation.
) else (
    echo Found model directory. Starting copy...
    
    rem Use robocopy to exclude model_cache
    rem /E :: copy subdirectories, including empty ones
    rem /XD :: exclude directories matching given names/paths
    rem /XO :: exclude older files (if re-running)
    robocopy "%MODEL_SRC%" "%MODEL_DST%" /E /XD "model_cache" /XO
    
    echo Robocopy finished with ErrorLevel: !errorlevel!
    
    rem Check if robocopy failed (exit code < 8 is success/partial success)
    if !errorlevel! gtr 7 (
        echo [ERROR] Robocopy failed with errorlevel !errorlevel!
        pause
        exit /b 1
    )
    
    set "ARCHIVE_NAME=%RELEASE_DIR%\ChickenRice_v2_Full.7z"
    if exist "%RELEASE_DIR%\ChickenRice_v2_Full.7z.*" del "%RELEASE_DIR%\ChickenRice_v2_Full.7z.*"
    
    echo Starting 7-Zip compression...
    rem Create split archive
    rem -v1800m: Split volume size 1800MB
    rem -mx9: Ultra compression
    "%SEVEN_ZIP%" a -t7z "!ARCHIVE_NAME!" "dist_release\ChickenRice_v2" -v1800m -mx9
    
    if errorlevel 1 (
        echo [ERROR] 7-Zip failed with errorlevel %errorlevel%
        pause
        exit /b 1
    )
    
    echo Created: !ARCHIVE_NAME!
    
    rem Cleanup large model from dist folder to save space/time for next run
    rem rmdir /s /q "%MODEL_DST%"
    echo (Model cleanup skipped for debugging)
)

echo.
echo ========================================
echo All Tasks Completed!
echo Releases are in: %RELEASE_DIR%
echo ========================================
pause
