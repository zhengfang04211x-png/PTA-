@echo off
chcp 65001 >nul
echo ========================================
echo PTA策略回测系统启动中...
echo ========================================
echo.

cd /d "%~dp0"

echo 正在检查Python环境...
python --version
if errorlevel 1 (
    echo 错误：未找到Python，请先安装Python！
    pause
    exit /b 1
)

echo.
echo 正在启动Streamlit应用...
echo.
echo 应用将在浏览器中自动打开
echo 如果没有自动打开，请手动访问: http://localhost:8501
echo.
echo 按 Ctrl+C 可以停止服务
echo ========================================
echo.

streamlit run app.py

pause
