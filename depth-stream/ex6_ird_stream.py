'''
https://github.com/elmonkey/Python_OpenNI2/tree/master/samples
Official primense openni2 and nite2 python bindings.
Streams infra-red camera
ref:
    http://www.eml.ele.cst.nihon-u.ac.jp/~momma/wiki/wiki.cgi/OpenNI/Python.html
@author: Carlos Torres <carlitos408@gmail.com>
'''

import cv2

from primesense import openni2#, nite2
import numpy as np
from primesense import _openni2 as c_api

#import matplotlib.pyplot as plt


## Directory where OpenNI2.so is located
#dist = '/home/carlos/Install/kinect/OpenNI2-Linux-Arm-2.2/Redist/'
dist ='/home/simon/deeplearning/openni/Install/kinect/openni2/OpenNI2-x64/Redist'
## Initialize openni and check
openni2.initialize(dist)#'C:\Program Files\OpenNI2\Redist\OpenNI2.dll') # accepts the path of the OpenNI redistribution
if (openni2.is_initialized()):
    print("openNI2 initialized")
else:
    print("openNI2 not initialized")

#### initialize nite and check
##nite2.initialize()
##if (nite2.is_initialized()):
##    print "nite2 initialized"
##else:
##    print "nite2 not initialized"
#### ===============================


dev = openni2.Device.open_any()
print('Some Device Information')
print('\t', dev.get_sensor_info(openni2.SENSOR_DEPTH))
print('\t', dev.get_sensor_info(openni2.SENSOR_IR))
print('\t', dev.get_sensor_info(openni2.SENSOR_COLOR))
#ut = nite2.UserTracker(dev)

## streams
# Depth stream
depth_stream = dev.create_depth_stream()

# IR stream
ir_stream = dev.create_ir_stream()

## Set stream speed and resolution
#w = 640
w = 640
#w = 320
#h = 480
h = 480
#h = 240
#fps=30
fps = 30

## Set the video properties
#print 'Get b4 video mode', depth_stream.get_video_mode()
#depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX=w, resolutionY=h, fps=fps))
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=w, resolutionY=h, fps=fps))
depth_stream.set_mirroring_enabled(False)
#depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=640, resolutionY=480, fps=30))
#print 'Get after video mode', depth_stream.get_video_mode()
ir_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_GRAY16, resolutionX=w, resolutionY=h, fps=fps))
ir_stream.set_mirroring_enabled(False)

## Start the streams
depth_stream.start()
ir_stream.start()


def get_depth1():
    """
    Returns numpy ndarrays representing raw and ranged depth images.
    Outputs:
        depth:= raw depth, 1L ndarray, dtype=uint16, min=0, max=2**12-1
        d4d  := depth for dislay, 3L ndarray, dtype=uint8, min=0, max=255
    Note1:
        fromstring is faster than asarray or frombuffer
    Note2:
        depth = depth.reshape(120,160) #smaller image for faster response
                NEEDS default video configuration
        depth = depth.reshape(240,320) # Used to MATCH RGB Image (OMAP/ARM)
    """
    depth_frame = depth_stream.read_frame()
    depth = np.fromstring(depth_frame().get_buffer_as_uint16(),dtype=np.uint16).reshape(h,w)  # Works & It's FAST
    d4d = np.uint8(depth.astype(float) *255/ 2**12-1) # Correct the range. Depth images are 12bits
    #d4d = cv2.cvtColor(d4d,cv2.COLOR_GRAY2RGB)
    d4d = np.dstack((d4d,d4d,d4d)) # faster than cv2 conversion
    return depth, d4d
#get_depth


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
    dmap = np.fromstring(depth_stream.read_frame().get_buffer_as_uint16(),dtype=np.uint16).reshape(h,w)  # Works & It's FAST
    d4d = np.uint8(dmap.astype(float) *255/ 2**12-1) # Correct the range. Depth images are 12bits
    #d4d = cv2.cvtColor(d4d,cv2.COLOR_GRAY2RGB)
    d4d = np.dstack((d4d,d4d,d4d)) # faster than cv2 conversion
    return dmap, d4d
#get_depth

def get_ir():
    """
    Returns numpy ndarrays representing raw and ranged infra-red(IR) images.
    Outputs:
        ir    := raw IR, 1L ndarray, dtype=uint16, min=0, max=2**12-1
        ir4d  := IR for display, 3L ndarray, dtype=uint8, min=0, max=255
    """
    ir_frame = ir_stream.read_frame()
    ir_frame_data = ir_stream.read_frame().get_buffer_as_uint16()
    ir4d = np.ndarray((ir_frame.height, ir_frame.width),dtype=np.uint16, buffer = ir_frame_data).astype(np.float32)
    ir4d = np.uint8((ir4d/ir4d.max()) * 255)
    ir4d = cv2.cvtColor(ir4d,cv2.COLOR_GRAY2RGB)
    return ir_frame, ir4d
#get_ir

frame_idx = 0

## main loop
done = False
while not done:
    key = cv2.waitKey(1)
    if (key&255) == 27:
        done = True
    ## Read in the streams
    # Depth
    dmap,d4d = get_depth()
    # Infrared
    ir_frame, ir4d = get_ir()
    cv2.imshow("Depth||IR", np.hstack((d4d, ir4d)))
    cv2.imwrite('checkboard.png',ir4d)

    #cv2.imshow("Depth", d4d)
    #cv2.imshow("IR", ir4d)
    frame_idx+=1
# end while

## Release resources and terminate
cv2.destroyAllWindows()
depth_stream.stop()
openni2.unload()
print ("Terminated")
