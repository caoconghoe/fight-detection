@echo off
rem Bam dup de chay he thong giam sat. Co the truyen nguon: run.bat 0 | run.bat http://<ip>:8080/video | run.bat video.mp4
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [LOI] Chua co moi truong .venv - xem README.md muc "Cai dat".
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"

if not exist "realtime\models\yolov8n-pose_openvino_model\yolov8n-pose.xml" (
    echo Chua co mo hinh YOLO OpenVINO -^> dang tao lan dau ^(tai yolov8n-pose.pt + export, mat 1-2 phut^)...
    python realtime\tools\benchmark.py --backend openvino --frames 30
)

python realtime\main.py %*
if errorlevel 1 pause
