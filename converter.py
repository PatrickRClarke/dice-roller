
from PIL import Image
import torchvision.transforms as transforms
import torch

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
])

def process_image(path):
    image = Image.open(path)
    image = transform(image)
    return image

image_tensor = process_image("test.jpg")
print(image_tensor.shape)


"""
First attempt

image = Image.open("test.jpg")

# L -> 8 Bit Grayscale
gray_image = image.convert("L")

# Resize to MNIST dimensions 
resized_image = gray_image.resize((28,28))

image_array = np.array(resized_image) / 255.0

resized_image.save("converted.jpg")
"""

