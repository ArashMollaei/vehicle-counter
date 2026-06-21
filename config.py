from pathlib import Path

# path
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_VIDEO = 'processed_output.mp4'

# model
MODEL_NAME = 'yolo11n.pt'

VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

DEFAULT_LINE_RATIO = 0.75

TRACKER_CONFIG = 'bytetrack.yaml'