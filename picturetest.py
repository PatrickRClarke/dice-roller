from picamera2 import Picamera2
from time import sleep

camera = Picamera2()



#camera.start()
#sleep(2)

###
#5/4/'25 - Attempting to improve resolution (originally 640 x 480 pixels)
capture_config = camera.create_still_configuration({"size":(3280,2464)})
camera.configure(capture_config)

full_res = camera.sensor_resolution
#print("Sensor resolution:", full_res)
# It is 3280, 2464
zoom_width  = full_res[0] // 7
zoom_height = full_res[1] // 7


#x = (full_res[0] - zoom_width) // 3
#y = (full_res[1] - zoom_height)// 3
#x = full_res[0] // 1
#y = full_res[1] // 1
x=1650
y=950

camera.set_controls({"ScalerCrop": (x, y, zoom_width, zoom_height)})

camera.start()
sleep(2)
###

camera.capture_file("test.jpg")

camera.stop()