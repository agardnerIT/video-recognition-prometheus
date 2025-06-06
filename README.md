# video-recognition-prometheus

<img width="638" alt="screenshot" src="https://github.com/user-attachments/assets/b2f7a062-34dd-4b9d-8fe6-a1fe3e16165f" />


Live Webcam Video Recognition of People with Prometheus Metrics.

This program uses your webcam to detect and display bounding boxes around humans. The application produces Prometheus metrics tracking the active number of detected people.

```
# HELP number_of_people_detected Current count of people detected
# TYPE number_of_people_detected gauge
number_of_people_detected 1.0
```

## Install & Run

You will need a webcam available. You will need to adjust the `--camera-index` to find the correct webcam. Start at `0` and work upwards until you find the correct camera.

```
python -m venv .
pip install -r requirements.txt
python webcamdetect.py --camera-index=0
```

## Parameters

* `--camera-index` (see above)
* `--sleep-time-between-updates` - (optional) Time in seconds. How often should the screen refresh and objects be identified? Defaults to `0.5`
* `--minimum-confidence-required` - (optional) How confident does the detection need to be to be displayed and tracked? Defaults to `0.7` (70% certainty)
* `--model` - (optional) Which model should we use to detect? Defaults to `yolov8n`
* `--hide-live-feed` (optional) Hides the live video feed window
* `--prometheus-port` (optional) Set the port for Prometheus metrics. Defaults to `8000`

## Prometheus Metrics
Prometheus Metrics are available at `http://localhost:8000/`
The port is configurable via the `--prometheus-port` parameter.
