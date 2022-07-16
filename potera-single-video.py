from time import sleep
import cv2
import numpy
import sys
import easyocr
from datetime import datetime
import csv
import os.path
import ffmpeg # https://github.com/deezer/spleeter/issues/101#issuecomment-554627345
import argparse

defaultFileName = "potjera-isecena.mp4"

# Arguments
parser = argparse.ArgumentParser(description="Potera single video processor",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-srcdir", "--srcDirectory", help="directory where file is located", default="examples/potera")
parser.add_argument("-file", "--fileName", help="video file name to be processed", default=defaultFileName)
parser.add_argument("-o", "--output", help="directory for csv and debug data output", default="results")
parser.add_argument("-lang", "--language", help="ocr language, can be either rs_latin or rs_cyrillic", default="rs_cyrillic")
parser.add_argument("-csv", "--csvFileName", help="name for csv file", default="questions.csv")
parser.add_argument("-d", "--debugData", help="create frame image files for every image processed. note: can use up a lot of data space!", default="True")
args = parser.parse_args()
config = vars(args)

# Configuration ################################################################

srcDir = config['srcDirectory']
fileName = config['fileName']
filePath = "%s/%s" %(srcDir, config['fileName'])
directoryOutput = config['output']
csvFileName = config['csvFileName']
createDebugData = (config['debugData'] == 'True')

# OCR language (either latin or cyrillic, cannot do both at the same time)
ocrLanguage = config['language']

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
howManyGreenFramesToJumpPrelod = 5
howManyFramesToJumpAfterSuccess = 350
frameIterationStepModifier = 1

# CSV config
csvResultsFileLocation = "%s/%s" %(directoryOutput, csvFileName)
csvLogFileLocation = "%s/log-%s" %(directoryOutput, csvFileName)

csvDelimeter = ';'
csvResultsHeaders = ['question', 'answer', 'video_bitrate', 'resolution_height', 'resolution_width', 'filename', 'frameNumber']
csvLogHeaders = ['filename', 'found_questions_answers', 'fps', 'iteration_step', 'processing_duration']

# End of configuration ##############################################################################

def print_progress_bar(index, total, label, endlabel):
    n_bar = 50  # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label} {index}/{total} {endlabel}")
    sys.stdout.flush()

def get_bitrate(file):
    probe = ffmpeg.probe(file)
    video_bitrate = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    bitrate = int(int(video_bitrate['bit_rate']) / 1000)
    return bitrate

def get_fps(file):
    probe = ffmpeg.probe(file)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    fps_first_part = int(video_info['r_frame_rate'].split('/')[0])
    fps_second_part = int(video_info['r_frame_rate'].split('/')[1])
    fps = int(fps_first_part / fps_second_part)
    return fps

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
print("Single video file processing started of %s..." %filePath)

if not os.path.isdir(srcDir):
    print('Incorrect srcDirectory: \"%s\" Does directory exist?' %srcDir)
    print('Skipping...')
    sys.exit(1)

if not os.path.isdir(directoryOutput):
    print('Incorrect output directory: \"%s\" Does directory exist?' %directoryOutput)
    print('Skipping...')
    sys.exit(1)

if not os.path.isfile(filePath):
    print('File path is incorrect: \"%s\" Does file exist?' %filePath)
    print('Skipping...')
    sys.exit(1)

# Load EasyOCR trained models (en is fallback)
reader = easyocr.Reader(['en', ocrLanguage], gpu=False)

# Initialize csv if not exist
if not os.path.isfile(csvResultsFileLocation):
    with open(csvResultsFileLocation, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter =';')
        writer.writerow(csvResultsHeaders)

if not os.path.isfile(csvLogFileLocation):
    with open(csvLogFileLocation, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter =';')
        writer.writerow(csvLogHeaders)

# Load up video and obtain first frame
videoFile = cv2.VideoCapture(filePath)
success,originalFrame = videoFile.read()
videoFileFramesTotalLength = int(videoFile.get(cv2.CAP_PROP_FRAME_COUNT))

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

# Get video bitrate for debug purposes
bitrate = get_bitrate(filePath)

frameIndex = 0
videoAverageFps = get_fps(filePath)
print("FPS: %d" %videoAverageFps)

howManyFramesToIterateBy = int(frameIterationStepModifier * videoAverageFps)
print("Frame iteration step: %d" %howManyFramesToIterateBy)

numberOfFoundQuestionAnswerPair = 0

# Loop through all frames of the video
while success:

    # Show preview of processing... 
    processingPreviewThumbnail = cv2.resize(originalFrame, (0, 0), fx=0.4, fy=0.4).copy()
    cv2.imshow('Processing video...', processingPreviewThumbnail)
    key = cv2.waitKey(1)

    # Stats
    currentTime = 'Time: {}'.format(datetime.now() - start_time)
    print_progress_bar(frameIndex, videoFileFramesTotalLength, "Frames: ", currentTime)
    
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
            success,originalFrame = videoFile.read()
            continue
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
                debugFrameName = "%s/%s-%d-1-frame-contours.jpg" % (directoryOutput, fileName, frameIndex)
                cv2.imwrite(debugFrameName, debugCopy)
                debugFrameName = "%s/%s-%d-2-question.jpg" % (directoryOutput, fileName, frameIndex)
                cv2.imwrite(debugFrameName, questionRectangleImage)
                debugFrameName = "%s/%s-%d-3-answer.jpg" % (directoryOutput, fileName, frameIndex)
                cv2.imwrite(debugFrameName, answerRectangleImage)

            ocrQuestionList = reader.readtext(questionRectangleImage, detail = 0, paragraph=True)
            ocrQuestion = listToString(ocrQuestionList)
            ocrAnswerList = reader.readtext(answerRectangleImage, detail = 0, paragraph=True)
            ocrAnswer = listToString(ocrAnswerList)

            print('\nQuestion: %s' %ocrQuestion)
            print('Answer: %s' %ocrAnswer)
            
            numberOfFoundQuestionAnswerPair += 1

            with open(csvResultsFileLocation, 'a+', encoding='UTF8', newline='') as f:
                writer = csv.writer(f, delimiter =';')
                csvDataRow = [ocrQuestion, ocrAnswer, bitrate, imageHeight, imageWidth, filePath, frameIndex]
                writer.writerow(csvDataRow)

            # https://subscription.packtpub.com/book/application-development/9781788474443/1/ch01lvl1sec15/jumping-between-frames-in-video-files
            frameIndex += howManyFramesToJumpAfterSuccess
            print("Jump to %dth frame of %d" %(frameIndex, videoFileFramesTotalLength))
            if frameIndex >= videoFileFramesTotalLength:
                print("No more frames to process after frame jump...")
            videoFile.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
            skipFirstGreenFoundMaskFrames = True

    # Read new frame and continue with the loop
    frameIndex += howManyFramesToIterateBy
    videoFile.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
    success,originalFrame = videoFile.read()

end_time = datetime.now()

print('\nFound: %d question/answer frames' %numberOfFoundQuestionAnswerPair)
duration = format(end_time - start_time)
print('Duration: {}'.format(end_time - start_time))

print("Finished processing of %s." %filePath)
with open(csvLogFileLocation, 'a+', encoding='UTF8', newline='') as f:
    writer = csv.writer(f, delimiter =';')
    csvDataRow = [filePath, numberOfFoundQuestionAnswerPair, videoAverageFps, howManyFramesToIterateBy, duration]
    writer.writerow(csvDataRow)