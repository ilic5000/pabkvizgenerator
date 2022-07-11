from time import sleep
import cv2
import numpy
import sys
import easyocr
from datetime import datetime
import csv
import os.path

# Configuration ################################################################

createDebugData = False

fileName = 'examples/potjera-e1320-isecena-najkrace.mp4'

# OCR language (either latin or cyrillic, cannot do both at the same time)
ocrLanguage = 'rs_latin'
#ocrLanguage = 'rs_cyrillic'

# Found contours area size treshold 
percentageOfAreaThreshold = 0.0035

# Contours scale config
blueMaskScale = 1.01
greenMaskScale = 1.01

# Add height in px (up and down equally) for masks (applied during generation of cropped image, not visible on contour)
blueMaskHeightExpansion = 5
blueMaskWidthExpansion = 0

greenMaskHeightExpansion = 10
greenMaskWidthExpansion = 0

# HSV masks values
green_l_h = 31
green_l_s = 23
green_l_v = 0
green_u_h = 84
green_u_s = 255
green_u_v = 255

blue_l_h = 100
blue_l_s = 118
blue_l_v = 42
blue_u_h = 120
blue_u_s = 255
blue_u_v = 210

# When answer/question are found, jump frames in order to avoid multiple detection of the same question
# This can be done smarter, but this simple jump works just fine
howManyGreenFramesToJumpPrelod = 10
howManyFramesToJump = 450

# CSV config
csvFileLocation = 'results/questions.csv'
csvDelimeter = ';'
csvHeaders = ['question', 'answer', 'filename', 'frameNumber']

# End of configuration ##############################################################################

def print_progress_bar(index, total, label):
    n_bar = 50  # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label} ({index}/{total})")
    sys.stdout.flush()

def listToString(s):
    str1 = " "
    return (str1.join(s))

def scale_contour(cnt, scale):
    if scale == 1.0:
        return cnt

    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(numpy.int32)

    return cnt_scaled

def areAllPointsInsideSeekBorderArea(contour, seekAreaBorderHorizontalY, seekAreaBorderVerticalXLeft, seekAreaBorderVerticalXRight):
    result = True 
    n = contour.ravel() 
    i = 0
    for j in n :
        if(i % 2 == 0):
            x = n[i]
            y = n[i + 1]
            if y < seekAreaBorderHorizontalY or x < seekAreaBorderVerticalXLeft or x > seekAreaBorderVerticalXRight:
                result = False
                break
        
        i = i + 1

    return result

def calculateMinMaxPoints(imageHeight, imageWidth, contour):
    n = contour.ravel() 
    i = 0
    ymin = imageHeight
    ymax = 0
    xmin = imageWidth
    xmax = 0

    for j in n :
        if(i % 2 == 0):
            x = n[i]
            y = n[i + 1]

            if y < ymin:
                ymin = y

            if y > ymax:
                ymax = y

            if x < xmin:
                xmin = x

            if x > xmax:
                xmax = x
            
        i = i + 1
    return ymin,ymax,xmin,xmax

############### Start of processing

start_time = datetime.now()
print("Started processing of %s..." %fileName)

# Load EasyOCR trained models
reader = easyocr.Reader([ocrLanguage], gpu=False)

# Initialize csv if not exist
if not os.path.isfile(csvFileLocation):
    with open(csvFileLocation, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter =';')
        writer.writerow(csvHeaders)

# Load up video and obtain first frame
videoFile = cv2.VideoCapture(fileName)
success,originalFrame = videoFile.read()
videoFileFramesTotalLength = int(videoFile.get(cv2.CAP_PROP_FRAME_COUNT))
frameIndex = 0

# Create seek area (a lot easier to find shapes and avoid false detections on unimportant parts of the image)
imageHeight, imageWidth, _ = originalFrame.shape 

seekAreaBorderHorizontalLineY = 2 * int(imageHeight/3)
seekAreaBorderHorizontalLineXStart = 0
seekAreaBorderHorizontalLineXEnd = imageWidth

seekAreaBorderLeftX = int(imageWidth/9.1)
seekAreaBorderLeftY = imageHeight

seekAreaBorderRightX = int(8.1 * int(imageWidth/9.1))
seekAreaBorderRightY = imageHeight

# Calculate area of found shapes tresholds
totalPixels = imageHeight * imageWidth
areaThreashold = percentageOfAreaThreshold * totalPixels

skipFirstGreenFoundMaskFrames = True

# Loop through all frames of the video
while success:
    
    print_progress_bar(frameIndex, videoFileFramesTotalLength, "Frames processed " + fileName)
    
    hsvFrameImage = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2HSV)
    
    # Create HSV masks 
    green_lower_hsv = numpy.array([green_l_h, green_l_s, green_l_v])
    green_upper_hsv = numpy.array([green_u_h, green_u_s, green_u_v])
    
    green_mask = cv2.inRange(hsvFrameImage, green_lower_hsv, green_upper_hsv)

    blue_lower_hsv = numpy.array([blue_l_h, blue_l_s, blue_l_v])
    blue_upper_hsv = numpy.array([blue_u_h, blue_u_s, blue_u_v])
    
    blue_mask = cv2.inRange(hsvFrameImage, blue_lower_hsv, blue_upper_hsv)

    # Erode green mask
    kernelGreen = numpy.ones((5,5), numpy.uint8)
    green_mask = cv2.erode(green_mask, kernelGreen)

    # Erode blue mask
    kernelBlue = numpy.ones((5,5), numpy.uint8)
    blue_mask = cv2.erode(blue_mask, kernelBlue)

    # Find contours in masked images
    contoursInGreenMask, _ = cv2.findContours(green_mask, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
    contoursInBlueMask, _ = cv2.findContours(blue_mask, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)

    # Magic! 

    maxGreenArea = 0 
    maxGreenAreaContour = None
    maxGreenAreaContourApprox = None
    green_ymin, green_ymax, green_xmin, green_xmax = None, None, None, None

    for cnt in contoursInGreenMask:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        numberOfPoints = len(approx)

        if area > maxGreenArea and numberOfPoints >= 4 and numberOfPoints <= 4 and area > areaThreashold and areAllPointsInsideSeekBorderArea(approx, seekAreaBorderHorizontalLineY, seekAreaBorderLeftX, seekAreaBorderRightX):
                green_ymin, green_ymax, green_xmin, green_xmax = calculateMinMaxPoints(imageHeight, imageWidth, approx)
                maxGreenArea = area
                maxGreenAreaContour = scale_contour(cnt, greenMaskScale)
                maxGreenAreaContourApprox = scale_contour(approx, greenMaskScale)
    
    maxBlueArea = 0 
    maxBlueAreaContour = None
    maxBlueAreaContourApprox = None
    blue_ymin, blue_ymax, blue_xmin, blue_xmax = None, None, None, None

    for cnt in contoursInBlueMask:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        numberOfPoints = len(approx)
        
        if area > maxBlueArea and numberOfPoints >= 4 and numberOfPoints <= 6 and area > areaThreashold and area > 3 * maxGreenArea and areAllPointsInsideSeekBorderArea(approx, seekAreaBorderHorizontalLineY, seekAreaBorderLeftX, seekAreaBorderRightX):
                blue_ymin, blue_ymax, blue_xmin, blue_xmax = calculateMinMaxPoints(imageHeight, imageWidth, approx) 
                maxBlueArea = area
                maxBlueAreaContour = scale_contour(cnt, blueMaskScale)
                maxBlueAreaContourApprox = scale_contour(approx, blueMaskScale)

    # Answer and question are found!
    if maxGreenArea > 0 and maxBlueArea > 0:
        if skipFirstGreenFoundMaskFrames:
            frameIndex += howManyGreenFramesToJumpPrelod
            videoFile.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
            skipFirstGreenFoundMaskFrames = False
        else:
            blue_ymin = blue_ymin - blueMaskHeightExpansion
            blue_ymax = blue_ymax + blueMaskHeightExpansion
            green_ymin = green_ymin - blueMaskHeightExpansion
            green_ymax = green_ymax + blueMaskHeightExpansion

            blue_xmin = blue_xmin - blueMaskWidthExpansion
            blue_xmax = blue_xmax + blueMaskWidthExpansion
            green_xmin = green_xmin - greenMaskWidthExpansion
            green_xmax = green_xmax + greenMaskWidthExpansion

            questionRectangleImage = originalFrame[blue_ymin:blue_ymax, blue_xmin:blue_xmax]
            answerRectangleImage = originalFrame[green_ymin:green_ymax, green_xmin:green_xmax]

            if createDebugData:
                cv2.imwrite("results/%s-%d-0-frame.jpg" % (fileName, frameIndex), originalFrame)
                debugCopy = originalFrame.copy()
                cv2.drawContours(debugCopy, [maxGreenAreaContourApprox], 0, (0, 255, 0), 3)
                cv2.drawContours(debugCopy, [maxBlueAreaContourApprox], 0, (255, 0, 0), 3)
                cv2.imwrite("results/%s-%d-1-frame-contours.jpg" % (fileName, frameIndex), debugCopy)

            cv2.imwrite("results/%s-%d-2-question.jpg" % (fileName, frameIndex), questionRectangleImage)
            ocrQuestionList = reader.readtext(questionRectangleImage, detail = 0, paragraph=True)
            ocrQuestion = listToString(ocrQuestionList)

            cv2.imwrite("results/%s-%d-3-answer.jpg" % (fileName, frameIndex), answerRectangleImage)
            ocrAnswerList = reader.readtext(answerRectangleImage, detail = 0, paragraph=True)
            ocrAnswer = listToString(ocrAnswerList)

            print('\nQuestion: %s' %ocrQuestion)
            print('Answer: %s' %ocrAnswer)

            with open(csvFileLocation, 'a+', encoding='UTF8', newline='') as f:
                writer = csv.writer(f, delimiter =';')
                csvDataRow = [ocrQuestion, ocrAnswer, fileName, frameIndex]
                writer.writerow(csvDataRow)

            # https://subscription.packtpub.com/book/application-development/9781788474443/1/ch01lvl1sec15/jumping-between-frames-in-video-files
            frameIndex += howManyFramesToJump
            print("Jump to %dth frame of %d" %(frameIndex, videoFileFramesTotalLength))
            if frameIndex >= videoFileFramesTotalLength:
                print("No more frames to process after frame jump...")
            videoFile.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
            skipFirstGreenFoundMaskFrames = True

    # Read new frame and continue with the loop
    success,originalFrame = videoFile.read()
    frameIndex += 1

end_time = datetime.now()
print('\nDuration: {}'.format(end_time - start_time))
print("Finished processing of %s." %fileName)