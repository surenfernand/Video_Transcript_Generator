@echo off
setlocal
cd /d "%~dp0."
"C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe" "%~dp0transcriber_app.py"
if errorlevel 1 (
    echo.
    echo The transcriber stopped because of an error.
    pause
)
endlocal
