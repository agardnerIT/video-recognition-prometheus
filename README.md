# video-recognition-prometheus
Live Webcam Video Recognition of People with Prometheus Metrics

### Install & Run

You will need a webcam available. You will need to adjust the `--camera-index` to find the correct webcam. Start at `0` and work upwards until you find the correct camera.

```
python -m venv .
pip install -r requirements.txt
python webcamdetect.py --camera-index=0 --sleep-time-between-updates=0.1 --minimum-confidence-required=0.4 --model=yolov8n
```
