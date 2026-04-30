
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from torchvision import transforms
from torchvision.models import vit_b_16
from ultralytics import YOLO


class AdaptiveWeatherDetectorBundle:
    def __init__(self, bundle_path, device=None):
        self.bundle_path = Path(bundle_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not self.bundle_path.exists():
            raise FileNotFoundError(f"Не найден bundle: {self.bundle_path}")

        self.bundle = torch.load(self.bundle_path, map_location="cpu", weights_only=False)
        self.meta = self.bundle["meta"]
        self.weather_classes = self.meta["weather_classes"]
        self.route_map = self.meta["route_map"]

        self._tmp_dir = Path(tempfile.mkdtemp(prefix="adaptive_bundle_"))

        self.classifier_ckpt = self.bundle["classifier_checkpoint"]
        cfg = self.classifier_ckpt.get("config", {})
        self.cls_img_size = int(cfg.get("img_size", 384))

        self.classifier = self._load_classifier(self.classifier_ckpt)
        self.detectors = self._load_detectors(self.bundle["experts"])

        self.cls_tfms = transforms.Compose([
            transforms.Resize((self.cls_img_size, self.cls_img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_classifier(self, ckpt):
        class_names = ckpt["class_names"]
        num_classes = len(class_names)

        cfg = ckpt.get("config", {})
        img_size = int(cfg.get("img_size", 384))

        model = vit_b_16(weights=None, image_size=img_size)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

        state = ckpt.get("model_state", ckpt.get("state_dict"))
        if state is None:
            raise RuntimeError("В classifier checkpoint нет model_state/state_dict")

        model.load_state_dict(state, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def _load_detectors(self, experts_dict):
        detectors = {}
        for expert_name, raw_bytes in experts_dict.items():
            pt_path = self._tmp_dir / f"{expert_name}.pt"
            pt_path.write_bytes(raw_bytes)
            detectors[expert_name] = YOLO(str(pt_path))
        return detectors

    def _to_pil(self, image):
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image).convert("RGB")
            return Image.fromarray(image.astype(np.uint8)).convert("RGB")
        raise TypeError("image must be str, Path, PIL.Image, or np.ndarray")

    @torch.no_grad()
    def predict_weather(self, image):
        pil_img = self._to_pil(image)
        x = self.cls_tfms(pil_img).unsqueeze(0).to(self.device)

        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())

        pred_weather = self.weather_classes[idx]
        confidence = float(probs[idx].item())
        prob_dict = {
            self.weather_classes[i]: float(probs[i].item())
            for i in range(len(self.weather_classes))
        }

        return pred_weather, confidence, prob_dict, pil_img

    @torch.no_grad()
    def predict(self, image, conf=0.25, imgsz=768, weather_override=None):
        if weather_override is None:
            pred_weather, weather_conf, weather_probs, pil_img = self.predict_weather(image)
        else:
            pil_img = self._to_pil(image)
            pred_weather = str(weather_override).lower()
            weather_conf = 1.0
            weather_probs = {w: float(w == pred_weather) for w in self.weather_classes}

        if pred_weather not in self.route_map:
            raise RuntimeError(f"Нет маршрута для погоды: {pred_weather}")

        expert_name = self.route_map[pred_weather]
        detector = self.detectors[expert_name]

        yolo_result = detector.predict(
            source=pil_img,
            conf=conf,
            imgsz=imgsz,
            verbose=False
        )[0]

        detections = []
        if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
            boxes = yolo_result.boxes.xyxy.detach().cpu().numpy()
            scores = yolo_result.boxes.conf.detach().cpu().numpy()
            classes = yolo_result.boxes.cls.detach().cpu().numpy().astype(int)
            names = yolo_result.names

            for box, score, cls_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box.tolist()
                detections.append({
                    "class_id": int(cls_id),
                    "class_name": names[int(cls_id)],
                    "confidence": float(score),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                })

        return {
            "predicted_weather": pred_weather,
            "weather_confidence": weather_conf,
            "weather_probs": weather_probs,
            "expert_used": expert_name,
            "detections": detections,
            "raw_result": yolo_result,
        }
