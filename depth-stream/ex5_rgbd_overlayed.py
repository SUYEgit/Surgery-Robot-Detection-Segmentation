#!/usr/bin/env python
'''
Created on 19Jun2015
Stream rgb and depth video side-by-side using openni2 opencv-python (cv2). 
RGB is overlayed on top on readable-depth. In addition, streams are aligned, mirror-corrected, and synchronized.

Requires the following libraries:
    1. OpenNI-Linux-<Platform>-2.2 <Library and driver>
    2. primesense-2.2.0.30 <python bindings>
    3. Python 2.7+
    4. OpenCV 2.4.X

Current features:
    1. Convert primensense oni -> numpy
    2. Stream and display rgb || depth || rgbd overlayed
    3. Keyboard commands    
        press esc to exit
        press s to save current screen and distancemap
    4. Sync and registered depth & rgb streams
    5. Print distance to center pixel
    6. Masks and overlays rgb stream on the depth stream

NOTE: 
    1. On device streams:  IR and RGB streams do not work together
       Depth & IR  = OK
       Depth & RGB = OK
       RGB & IR    = NOT OK
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
dist ='/home/carlos/Install/openni2/OpenNI-Linux-x64-2.2/Redist'


## initialize openni and check
openni2.initialize(dist) #'C:\Program Files\OpenNI2\Redist\OpenNI2.dll') # accepts the path of the OpenNI redistribution
if (openni2.is_initialized()):
    print "openNI2 initialized"
else:
    print "openNI2 not initialized"

## Register the device
dev = openni2.Device.open_any()

## create the streams stream
rgb_stream = dev.create_color_stream()
depth_stream = dev.create_depth_stream()

##configure the depth_stream
#print 'Get b4 video mode', depth_stream.get_video_mode()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=320, resolutionY=240, fps=30))


## Check and configure the mirroring -- default is True
# print 'Mirroring info1', depth_stream.get_mirroring_enabled()
depth_stream.set_mirroring_enabled(False)
rgb_stream.set_mirroring_enabled(False)


## start the stream
rgb_stream.start()
depth_stream.start()

## synchronize the streams
dev.set_depth_color_sync_enabled(True) # synchronize the streams

## IMPORTANT: ALIGN DEPTH2RGB (depth wrapped to match rgb stream)
dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

##help(dev.set_image_registration_mode)


def get_rgb():
    """
    Returns numpy 3L ndarray to represent the rgb image.
    """
    bgr   = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(240,320,3)
    rgb   = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    return rgb    
#get_rgb




def get_depth():
    """
    Returns numpy ndarrays representing the raw and ranged depth images.
    Outputs:
        dmap:= distancemap in mm, 1L ndarray, dtype=uint16, min=0, max=2**12-1
        d4d := depth for dislay, 3L ndarray, dtype=uint8, min=0, max=255    
    Note1: 
        fromstring is faster than asarray or frombuffer
    Note2:     
        .reshape(120,160) #smaller image for faster response 
                OMAP/ARM default video configuration
        .reshape(240,320) # Used to MATCH RGB Image (OMAP/ARM)
                Requires .set_video_mode
    """
    dmap = np.fromstring(depth_stream.read_frame().get_buffer_as_uint16(),dtype=np.uint16).reshape(240,320)  # Works & It's FAST
    d4d = np.uint8(dmap.astype(float) *255/ 2**12-1) # Correct the range. Depth images are 12bits
    d4d = 255 - cv2.cvtColor(d4d,cv2.COLOR_GRAY2RGB)
    return dmap, d4d
#get_depth




def mask_rgbd(d4d,rgb, th=0):
    """
    Overlays images and uses some blur to slightly smooth the mask
    (3L ndarray, 3L ndarray) -> 3L ndarray
    th:= threshold
    """
    mask = d4d.copy()
    #mask = cv2.GaussianBlur(mask, (5,5),0)
    idx =(mask>th)
    mask[idx] = rgb[idx]
    return mask
#mask_rgbd


## main loop
s=0
done = False
while not done:
    key = cv2.waitKey(1) & 255
    ## Read keystrokes
    if key == 27: # terminate
        print "\tESC key detected!"
        done = True
    elif chr(key) =='s': #screen capture
        print "\ts key detected. Saving image and distance map {}".format(s)
        cv2.imwrite("ex5_"+str(s)+'.png', canvas)
        np.savetxt("ex5dmap_"+str(s)+'.out',dmap)
        #s+=1 # uncomment for multiple captures        
    #if
    
    ## Streams
    #RGB
    rgb = get_rgb()
    
    #DEPTH
    dmap,d4d = get_depth()
    
    # Overlay rgb over the depth stream
    rgbd  = mask_rgbd(d4d,rgb)
    
    # canvas
    canvas = np.hstack((d4d,rgb,rgbd))
    
    ## Distance map
    print 'Center pixel is {} mm away'.format(dmap[119,159])

    ## Display the stream
    cv2.imshow('depth || rgb || rgbd', canvas )
# end while

## Release resources 
cv2.destroyAllWindows()
rgb_stream.stop()
depth_stream.stop()
openni2.unload()
print ("Terminated")
