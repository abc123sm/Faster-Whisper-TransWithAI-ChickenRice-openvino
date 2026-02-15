@echo off
:: =================================================================
:: 这个批处理文件用于在安全的环境中运行模型转换脚本 (VENV 版本)
:: =================================================================

:: 1. 激活你的 VENV 环境
::    这个路径是根据你的项目结构猜测的。
::    如果你的 venv 文件夹不叫 "ChickenRice_v2"，请修改下面的路径。
echo Activating VENV environment...
call "C:\AI_zimu_jihua\code\ChickenRice_v2\venv\Scripts\activate.bat"

:: 检查环境是否激活成功
if not defined VIRTUAL_ENV (
    echo.
    echo ERROR: Failed to activate VENV environment.
    echo Please check if the path below is correct:
    echo "C:\AI_zimu_jihua\code\ChickenRice_v2\venv\Scripts\activate.bat"
    pause
    exit /b
)

echo.
echo VENV environment activated successfully.
echo.
echo =================================================================
echo.

:: 2. 强制设置临时文件夹为纯英文路径
echo Setting temporary directory to C:\temp to avoid path encoding issues...
set TEMP=C:\AI_zimu_jihua\code\ChickenRice_v2\models\temp
set TMP=C:\AI_zimu_jihua\code\ChickenRice_v2\models\tmp

echo.
echo =================================================================
echo.

cd ChickenRice_v2
:: 3. 运行你的 Python 转换脚本
echo Starting model conversion...
echo.
python convert_model.py

:: 4. 保持窗口打开，以便查看结果
echo.
echo =================================================================
echo Script finished.
pause


