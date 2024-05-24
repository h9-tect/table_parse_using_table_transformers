import torch
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection

def load_detection_model():
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device

def load_structure_model(device):
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
    model.to(device)
    return model
