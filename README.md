%%writefile README.md
# Adaptive Weather Object Detector

Inference-only demo of an adaptive object detection system for road scenes under adverse weather conditions.

The system analyzes an input image, predicts the visibility/weather condition, selects an appropriate expert route, and performs object detection.

## Features

- Image upload through a local Gradio interface
- Automatic weather/visibility classification
- Adaptive expert routing
- Object detection with bounding boxes
- Detection table with classes, confidence scores and coordinates
- Weather probability table
- Decision log explaining the selected route

## Supported weather classes

- fog
- rain
- sand
- snow

## Adaptive routing

| Weather condition | Expert route |
|---|---|
| rain | rain_snow |
| snow | rain_snow |
| fog | fog_sand |
| sand | fog_sand |

## Repository contents

| File | Description |
|---|---|
| `app.py` | Gradio web interface |
| `adaptive_infer.py` | Inference logic and adaptive model wrapper |
| `adaptive_detector_bundle.pt` | Model bundle with weather classifier and detection experts |
| `requirements.txt` | Python dependencies |

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/adaptive-weather-object-detector.git
cd adaptive-weather-object-detector
```

Install dependencies:
```bash
pip install -r requirements.txt
```
Run
```bash
python app.py
```
After launch, open the Gradio link in the browser.

## Usage
- Upload an image of a road scene.
- Select automatic weather detection or choose a weather class manually.
- Set the detection confidence threshold.
- Click Run analysis.
- View the annotated image, detected objects, weather probabilities and decision log.

### Inference-only notice

This repository contains only the inference/demo part of the project. Training scripts, datasets and experimental notebooks are not included.

License

This project is released under the MIT License.
