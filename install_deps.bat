@echo off
setlocal
echo ============================================================
echo   MYSTRAN Viewer -- Dependency Installer (Windows)
echo ============================================================
echo.

:: ── Check Python is available ─────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found on PATH.
    echo        Install Python 3.10+ from https://python.org
    echo        Tick "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: ── Print version ─────────────────────────────────────────────────────────────
echo Python found:
python --version
echo.

:: ── Choose mode ───────────────────────────────────────────────────────────────
echo Options:
echo   1  Install core packages only  (moderngl, glfw, pyrr, numpy, imgui-bundle)
echo   2  Install core + pyNastran    (adds OP2 / F06 file support)
echo   3  Check what is installed
echo.
set /p CHOICE="Enter choice [1/2/3, default=2]: "
if "%CHOICE%"=="" set CHOICE=2

echo.

if "%CHOICE%"=="3" (
    echo Checking installed packages...
    python install_deps.py --check
    goto END
)

if "%CHOICE%"=="2" (
    echo Installing core packages + pyNastran (OP2 support^)...
    python install_deps.py --with-op2
    goto CHECK_RESULT
)

echo Installing core packages...
python install_deps.py
goto CHECK_RESULT

:CHECK_RESULT
if errorlevel 1 (
    echo.
    echo One or more packages failed. See messages above.
    echo Troubleshooting:
    echo   - Make sure your GPU driver supports OpenGL 3.3+
    echo   - Try: pip install moderngl --no-cache-dir
    echo   - On older hardware, integrated Intel graphics may need a driver update.
    goto END
)

echo.
echo ============================================================
echo   Ready!  To launch the viewer:
echo     python main.py
echo   To open a model directly:
echo     python main.py "C:\path\to\model.bdf"
echo ============================================================

:END
echo.
pause
