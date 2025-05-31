# With thanks to
# - https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993
# - https://datalab.medium.com/yolov8-detection-from-webcam-step-by-step-cpu-d590a0700e36

from ultralytics import YOLO
import cv2
import math
import time
import argparse
import prometheus_client

# Create a Prometheus Gauge to store the current number of people detected
number_of_people_detected = prometheus_client.Gauge('number_of_people_detected', 'Current count of people detected')

def capture_and_categorise(camera_index, sleep_time_between_updates, minimum_confidence_required, model, hide_live_feed):
    # start webcam
    cap = cv2.VideoCapture(camera_index)
    # Set capture width and height
    cap.set(3, 640)
    cap.set(4, 480)

    # model
    # Note: The model file will be downloaded on first run
    # OR you can pre-download and store in the same folder as the script
    # If you want, pre-download from here:
    # https://github.com/ultralytics/assets/releases
    model = YOLO(model=model, task="detect", verbose=False)

    # object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"
                ]

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            # How many objects were detected in this frame?
            # Note: Not all of these will be people
            print(f"Detected: {len(boxes)} objects...")
            NUMBER_OF_PEOPLE = 0

            # For each detected object...
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100

                # class name
                cls = int(box.cls[0])
                class_name_human_readable = classNames[cls]

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                # Only paint bounding box
                # If confidence is greater than threshold
                # and the object is a person
                if confidence > minimum_confidence_required and class_name_human_readable == "person":
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, f"{class_name_human_readable} [{confidence}]", org, font, fontScale, color, thickness)

                    # Confident enough that this is a human?
                    # Increment the number of people by 1
                    NUMBER_OF_PEOPLE += 1

            # All done processing and painting
            # boxes for this frame
            # Set Prometheus metric to however many humans were found in image
            number_of_people_detected.set(value=NUMBER_OF_PEOPLE)

        if not hide_live_feed:
            cv2.imshow('Webcam - Live Feed (Ctrl+C or q to exit)', img)

        # Sleep between detections
        time.sleep(sleep_time_between_updates)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main(camera_index: int, sleep_time_between_updates: float = 0.5, minimum_confidence_required: float = 0.7, model: str = "yolov8n.pt", hide_live_feed: bool = False, prometheus_port: int = 8000):
    # Start the Prometheus endpoint first
    prometheus_client.start_http_server(port=prometheus_port)
    prometheus_client.REGISTRY.unregister(prometheus_client.PLATFORM_COLLECTOR)
    prometheus_client.REGISTRY.unregister(prometheus_client.PROCESS_COLLECTOR)
    prometheus_client.REGISTRY.unregister(prometheus_client.GC_COLLECTOR)
    # Then start the camera capture
    capture_and_categorise(camera_index=camera_index, sleep_time_between_updates=sleep_time_between_updates, minimum_confidence_required=minimum_confidence_required, model=model, hide_live_feed=hide_live_feed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--camera-index", help="Which camera should be used? Experiment with this value from 0 upwards", default=0, type=int)
    parser.add_argument("--sleep-time-between-updates", help="How long between screen captures. 1.0 = 1 frame per second", default=0.5, type=float)
    parser.add_argument("--model", help="Which model should be used? Defaults to yolov8n.pt", default="yolov8n.pt", type=str)
    parser.add_argument("--minimum-confidence-required", help="How confident does model need to be to categorise an object? Default 0.4 = 40%", default=0.4, type=float)
    parser.add_argument("--hide-live-feed", help="(optional) Visually display the camera window or not? Default false", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--prometheus-port", help="Port to serve prometheus metrics on. Default: 8000", default=8000, type=int)

    args = parser.parse_args()

    main(camera_index=args.camera_index,
        sleep_time_between_updates=args.sleep_time_between_updates,
        model=args.model,
        minimum_confidence_required=args.minimum_confidence_required,
        hide_live_feed=args.hide_live_feed,
        prometheus_port=args.prometheus_port)
