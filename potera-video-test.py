import cv2
import numpy
import sys

def print_progress_bar(index, total, label):
    n_bar = 50  # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}")
    sys.stdout.flush()

fileName = 'potjera-e1320-isecena-najkrace.mp4'
videoFile = cv2.VideoCapture(fileName)
success,originalFrame = videoFile.read()
length = int(videoFile.get(cv2.CAP_PROP_FRAME_COUNT))
frameIndex = 0

while success:
    print_progress_bar(frameIndex, length, "Frames processed " + fileName)
    
    hsvFrame = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2HSV)
  
    lower_green = numpy.array([121,47,57])
    upper_green = numpy.array([121,76,58])

    lower_red = numpy.array([5,84,100])
    upper_red = numpy.array([5,100,100])

    greenMask = cv2.inRange(hsvFrame, lower_green, upper_green)

    cv2.imwrite("frame%d.jpg" % frameIndex, originalFrame)     # save frame as JPEG file      
    cv2.imwrite("gframe%d.jpg" % frameIndex, greenMask)     # save frame as JPEG file     
  
    success,originalFrame = videoFile.read()
    #print('Read a new frame: ', success)
    frameIndex += 1
print("DONE!")