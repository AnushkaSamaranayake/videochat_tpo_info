import torch
from groundingdino.util.inference import Model as GroundingDINO
from sam2.build_sam import build_sam2
from cotracker.predictor import CoTrackerPredictor


class SegmentationHook:
    def __init__(self, device="cuda"):
        self.model = build_sam2("sam2_hiera_large.yaml")
        self.model.to(device)
        self.device = device

    def segment(self, image, box):
        with torch.no_grad():
            masks = self.model.predict(image=image, boxes=box)
        return masks


class DetectionHook:
    def __init__(self, device="cuda"):
        self.model = GroundingDINO(model_config_path="GroundingDINO_SwinT_OGC.py",
                                   model_checkpoint_path="groundingdino_swint_ogc.pth")
        self.model.to(device)
        self.device = device

    def detect(self, image, text_prompt):
        boxes, logits = self.model.predict(image, text_prompt)
        return boxes, logits


class TrackingHook:
    def __init__(self, device="cuda"):
        self.model = CoTrackerPredictor()
        self.model.to(device)
        self.device = device

    def track(self, video_tensor, points):
        with torch.no_grad():
            tracks = self.model(video_tensor, query_points=points)
        return tracks