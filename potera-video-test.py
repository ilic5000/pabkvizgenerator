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
  
    lower_green = numpy.array([31,23,0])
    upper_green = numpy.array([84,255,255])

    greenMask = cv2.inRange(hsvFrame, lower_green, upper_green)

    # Erode mask
    kernel = numpy.ones((5,5), numpy.uint8)
    greenMask = cv2.erode(greenMask, kernel)

    contours, _ = cv2.findContours(greenMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        #print('len approx %d' %len(approx))
        #print('area %d' %area)
        if len(approx) > 4:
            if area > 5000:
                #cv2.drawContours(original_img_preview, [approx], 0, (0, 0, 255), 1)
                cv2.imwrite("results/frame%d.jpg" % frameIndex, originalFrame)     # save frame as JPEG file      
                #cv2.imwrite("results/mframe%d.jpg" % frameIndex, greenMask)     # save frame as JPEG file     
                
                # https://subscription.packtpub.com/book/application-development/9781788474443/1/ch01lvl1sec15/jumping-between-frames-in-video-files
                howManyFramesToJump = 450
                videoFile.set(cv2.CAP_PROP_POS_FRAMES, howManyFramesToJump)
                frameIndex += howManyFramesToJump
  
    success,originalFrame = videoFile.read()
    #print('Read a new frame: ', success)
    frameIndex += 1
print("DONE!")