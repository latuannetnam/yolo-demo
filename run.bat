@echo off
echo Starting Real-time Object Detection System...
echo.
echo Controls:
echo - Press 'q' or ESC to quit
echo - Press 's' to save screenshots
echo - Press 'r' to reset statistics
echo.
uv run python main_object_detection.py
echo.
echo Application finished. Press any key to exit...
pause >nul
