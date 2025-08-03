import RPi.GPIO as GPIO
import time

# Configure pin and pull-down resistor.
GPIO.setmode(GPIO.BCM)

input_pin = 17
GPIO.setup(input_pin, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)

output_pin = 27
GPIO.setup(output_pin, GPIO.OUT)

try: 
    while True:
        if GPIO.input(input_pin) == GPIO.HIGH:
            print("The red button has been pushed! What is wrong with you?!")
            GPIO.output(output_pin, GPIO.HIGH)
            # I will need to experiment to see how long it takes for the sphere to spin.
            time.sleep(1)
            GPIO.output(output_pin, GPIO.LOW)
            
        else:
            print(time.strftime("%H:%M:%S", time.localtime() ) )
        time.sleep(0.5)
        
except KeyboardInterrupt:
    print("Exiting program... I hope you're happy.")
    
finally:
    GPIO.cleanup()
    
        

