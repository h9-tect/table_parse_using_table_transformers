from torchvision import transforms
from PIL import Image

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        return resized_image

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def prepare_image(image, device):
    pixel_values = detection_transform(image).unsqueeze(0)
    pixel_values = pixel_values.to(device)
    return pixel_values

def prepare_cropped_image(cropped_image, device):
    pixel_values = structure_transform(cropped_image).unsqueeze(0)
    pixel_values = pixel_values.to(device)
    return pixel_values

def outputs_to_objects(outputs, image_size, id2label):
    objects = []
    for output in outputs:
        """
        ensure that each output in the outputs list is a dictionary
        and contains the necessary keys 'score', 'class' 'box'
        """
        if isinstance(output, dict) and 'score' in output and 'class' in output and 'box' in output:
            if output['score'] > 0.5:  # filter objects with with low confidence scores
                label = id2label[output['class']]
                x, y, w, h = output['box'] # counding box coordinates
                # converting from rel vals to pixel ones
                x *= image_size[0]
                w *= image_size[0]
                y *= image_size[1]
                h *= image_size[1]
                # detected objects with their labels and coordinates
                objects.append({
                    'label': label,
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                })
    return objects

