DRIVERSENSE — ADVANCED DRIVER DROWSINESS DETECTION DASHBOARD
============================================================

1. REQUIREMENTS
- Python 3.13
- Windows webcam
- models/face_landmarker.task (your existing MediaPipe Face Landmarker model)

2. PROJECT STRUCTURE
Driver_Drowsiness_Detection_Advanced/
  app.py
  requirements.txt
  README.txt
  models/
    face_landmarker.task   <-- COPY YOUR EXISTING MODEL HERE
  logs/                    <-- generated CSV and PDF reports
  templates/
    index.html
  static/
    style.css
    script.js

3. SETUP IN WINDOWS CMD
Open CMD in this project folder:

    py -3.13 -m venv drowsy_env
    drowsy_env\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

If you already have a working drowsy_env, activate it and run:
    pip install flask reportlab

4. MODEL FILE
Copy your existing file:
    models\face_landmarker.task

into the models folder in this project.

5. RUN
    drowsy_env\Scripts\activate
    python app.py

Open:
    http://127.0.0.1:5000

Close any other application using the webcam before starting.

6. FEATURES
- Live webcam monitoring through Flask/MJPEG
- MediaPipe Tasks Face Landmarker
- EAR eye closure detection
- MAR-based possible yawn detection
- Rolling 60-second PERCLOS
- Drowsiness alert and Windows beep
- Live EAR/MAR/PERCLOS graph
- Session history
- CSV event and session exports
- PDF session report

7. IMPORTANT CALIBRATION NOTES
EAR_THRESHOLD=0.23
DROWSY_SECONDS=2.0
MAR_THRESHOLD=0.55
YAWN_SECONDS=1.2
PERCLOS_WINDOW=60.0

These are experimental project parameters. MAR can be affected by speech, camera angle,
lighting and individual facial geometry. Calibrate thresholds using your own camera.
PERCLOS here is computed from valid face-detected samples in the rolling window.

8. SAFETY
This is an educational prototype and must not be used as a certified driver safety,
medical, or real-world vehicle control system.
