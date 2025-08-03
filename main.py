# Patrick Clarke
# CECS456 - Machine Learning
# Final Project
# 4/19/'25

# Camera
from picamera2 import Picamera2
from time import sleep

camera = Picamera2()

#GPIO
import RPi.GPIO as GPIO
import time

# Configure pin and pull-down resistor.
GPIO.setmode(GPIO.BCM)

input_pin = 17
GPIO.setup(input_pin, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)

output_pin = 27
GPIO.setup(output_pin, GPIO.OUT)


# Model
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision.transforms import ToTensor



test_data = datasets.MNIST (
    
    root = 'data',
    train = False,
    transform = ToTensor(),
    #download = True
    download = False
)

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        
        return F.softmax(x)


# In my case it won't be available because I am using a Raspberry Pi 4B.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the training model.
model = CNN().to(device)
model.load_state_dict(torch.load('Dice_Model.pth'))

model.eval()

#I'm not sure what the point of the target is.
data, target = test_data[10]

# Test
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


# End Test

# Optimizing the digital zoom for the camera
capture_config = camera.create_still_configuration({"size":(3280,2464)})
camera.configure(capture_config)

full_res = camera.sensor_resolution

zoom_width  = full_res[0] // 7
zoom_height = full_res[1] // 7

x=1650
y=950

camera.set_controls({"ScalerCrop": (x, y, zoom_width, zoom_height)})

try: 
    while True:
        if GPIO.input(input_pin) == GPIO.HIGH:
            print("The red button has been pushed! What is wrong with you?!")
            GPIO.output(output_pin, GPIO.HIGH)
            # I will need to experiment to see how long it takes for the sphere to spin.
            time.sleep(1)
            GPIO.output(output_pin, GPIO.LOW)
            
            # Actually taking the picture.
            camera.start()
            sleep(2)
            camera.capture_file("test.jpg")
            camera.stop()

            image_tensor = process_image("test.jpg")
            data=image_tensor

            data = data.unsqueeze(0).to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True).item()
            print(f'Prediction: {prediction}')
            image = data.squeeze(0).squeeze(0).cpu().numpy()

            plt.imshow(image, cmap='gray')
            plt.show()
            
        else:
            print(time.strftime("%H:%M:%S", time.localtime() ) )
        time.sleep(0.5)
        
except KeyboardInterrupt:
    print("Exiting program... I hope you're happy.")
    
finally:
    GPIO.cleanup()
