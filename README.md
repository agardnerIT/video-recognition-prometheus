# video-recognition-prometheus
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
python webcamdetect.py --camera-index=0 --sleep-time-between-updates=0.1 --minimum-confidence-required=0.4 --model=yolov8n --h
ide-live-feed
```

## Parameters

* `--camera-index` (see above)
* `--sleep-time-between-updates` - Time in seconds. How often should the screen refresh and objects be identified? Defaults to `0.5`
* `--minimum-confidence-required` - How confident does the detection need to be to be displayed and tracked?
* `--model` - (optional) Which model should we use to detect? Defaults to `yolov8n`
* `--hide-live-feed` (optional) Hides the live video feed window

## Prometheus Metrics
Prometheus Metrics are available at `http://localhost:8000/`
The port is configurable via the `--
