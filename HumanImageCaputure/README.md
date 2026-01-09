# Human Image Capture Service

Detects persons via webcam and sends cropped images through OutputLayer.

## Installation

```bash
pip install -r requirements.txt
```

Download YOLOE model: `yoloe-11s-seg.pt`


## Configuration

Edit `CONFIG` dictionary in the script:

- `confidence` (0.15): Detection threshold
- `frames_to_confirm` (20): Frames needed to confirm a person
- `frames_to_lose` (100): Frames until person is lost
- `position_tolerance` (150): Max distance in pixels for tracking
- `camera_index` (0): Webcam ID
- `export_dir` ("detections"): Output directory for saved images

## Output

- Cropped person images saved to `detections/`
- Images sent via OutputLayer with metadata

