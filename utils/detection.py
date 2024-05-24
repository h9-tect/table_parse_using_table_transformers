import torch

def detect_tables(model, pixel_values):
    with torch.no_grad():
        outputs = model(pixel_values)
    return outputs

def detect_cells(model, pixel_values):
    with torch.no_grad():
        outputs = model(pixel_values)
    return outputs
