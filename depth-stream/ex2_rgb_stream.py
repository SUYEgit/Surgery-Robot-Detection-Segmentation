#!/usr/bin/env python
'''
Created on 19Jun2015
Stream rgb video using openni2 opencv-python (cv2)

Requires the following libraries:
    1. OpenNI-Linux-<Platform>-2.2 <Library and driver>
    2. primesense-2.2.0.30 <python bindings>
    3. Python 2.7+
    4. OpenCV 2.4.X

Current features:
    1. Convert primensense oni -> numpy
    2. Stream and display rgb
    3. Keyboard commands    
        press esc to exit
        press s to save current screen


NOTE: 
    1. On device streams:  IR and RGB streams do not work together
       Depth & IR  = OK
       Depth & RGB = OK
       RGB & IR    = NOT OK

    2. Do not synchronize with depth or stream will feeze
    
@author: Carlos Torres <carlitos408@gmail.com>
'''

import numpy as np
import cv2
from primesense import openni2#, nite2
from primesense import _openni2 as c_api

## Path of the OpenNI redistribution OpenNI2.so or OpenNI2.dll
# Windows
#dist = 'C:\Program Files\OpenNI2\Redist\OpenNI2.dll'
# OMAP
#dist = '/home/carlos/Install/kinect/OpenNI2-Linux-ARM-2.2/Redist/'
# Linux
dist ='/home/simon/deeplearning/openni/Install/kinect/openni2/OpenNI2-x64/Redist'

## Initialize openni and check
openni2.initialize(dist) #
if (openni2.is_initialized()):
    print ("openNI2 initialized")
else:
    print ("openNI2 not initialized")

## Register the device
dev = openni2.Device.open_any()

## Create the streams stream
rgb_stream = dev.create_color_stream()

## Check and configure the depth_stream -- set automatically based on bus speed
print ('The rgb video mode is', rgb_stream.get_video_mode())   # Checks rgb video configuration
rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=320, resolutionY=240, fps=30))

## Start the streams
rgb_stream.start()

## Use 'help' to get more info
# help(dev.set_image_registration_mode)


def get_rgb():
    """
    Returns numpy 3L ndarray to represent the rgb image.
    """
    bgr   = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(240,320,3)
    rgb   = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    return rgb    
#get_rgb


## main loop
s=0
done = False
while not done:
    key = cv2.waitKey(1) & 255
    ## Read keystrokes
    if key == 27: # terminate
        print ("\tESC key detected!")
        done = True
    elif chr(key) =='s': #screen capture
        print ("\ts key detected. Saving image {}".format(s))
        cv2.imwrite("ex2_"+str(s)+'.png', rgb)
        #s+=1 # uncomment for multiple captures
    #if
    
    ## Streams    
    #RGB
    rgb = get_rgb()

    ## Display the stream syde-by-side
    cv2.imshow('rgb', rgb)
# end while

## Release resources 
cv2.destroyAllWindows()
rgb_stream.stop()
openni2.unload()
print ("Terminated")
