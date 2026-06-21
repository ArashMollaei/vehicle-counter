import cv2
import logging
from typing import Dict, Tuple
from collections import defaultdict

from ultralytics import YOLO

from config import MODEL_NAME, VEHICLE_CLASSES, DEFAULT_LINE_RATIO, TRACKER_CONFIG, OUTPUT_VIDEO

logging.basicConfig(level= logging.INFO)
logger = logging.getLogger(__name__)


class VehicleCounter:
    
    def __init__(self, model_name: str= MODEL_NAME, line_ratio: float= DEFAULT_LINE_RATIO):
        """
        Args:
            model_name: YOLO model name (from pre-trained or custom path)
            line_ratio: Horizontal count line position (height ratio)
        """
        self.line_ratio = line_ratio
        self.model = self._load_model(model_name)
        self.counted_ids : Dict[int, set] = defaultdict(set)
    
    def _load_model(self, model_name: str) -> YOLO:
        """
        Model loading with error handling.
        """
        
        try:
            model = YOLO(model_name)
            logger.info(f'Model {model_name} loaded successfully.')
            return model
        
        except Exception as e:
            logger.error(f'Failed to load model {model_name}: {e}')
            raise RuntimeError(f'Could not load YOLO model: {e}')
        
    def process_video(self, video_path: str, output_path: str = OUTPUT_VIDEO) ->  Dict[Tuple[int, int], str]:
        """
        Video processing: tracking, counting, and storing output video.
        
        Returns: 
            tuple: Dictionary of counting cars by class and output video path
        
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f'Cannot open video file: {video_path}')

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output video settings 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError('Cannot initialize video writer.')
        
        line_y = int(height * self.line_ratio)
        
        self.counted_ids.clear()
        frame_count = 0
        
        logger.info('Processing video ...')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            
            # Tracking (vehicle classes only)
            results = self.model.track(
                frame,
                persist= True,
                classes = list(VEHICLE_CLASSES.keys()),
                tracker = TRACKER_CONFIG,
                verbose = False
            )
            
            annotated_frame = results[0].plot()
            
            # Draw a counting line
            cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 255, 0), 2)
            
            # Extract tracking information
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                class_ids = results[0].boxes.cls.int().cpu().numpy()
                
                for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                    
                    center_y = box[1] 
                    # Count if the center crosses the line and has not been counted before
                    if center_y > line_y and track_id not in self.counted_ids[cls_id]:
                        self.counted_ids[cls_id].add(track_id)
            
            out.write(annotated_frame)
            frame_count += 1
            if frame_count % 50 == 0:
                logger.info(f'Processed {frame_count} / {total_frames} frames.')
        
        cap.release()
        out.release()
        logger.info('Processing finished.')
        
        
        # Converting counts into a final dictionary
        counts = {cls: len(ids) for cls, ids in self.counted_ids.items()}
        return counts, output_path