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

# Arguments
parser = argparse.ArgumentParser(description="Slagalica single video processor",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-srcdir", "--srcDirectory", help="directory where file is located", default="examples")
parser.add_argument("-file", "--fileName", help="video file name to be processed", default="slagalica_nova_isecena.mp4")
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
    currentTime = 'Time: {}'.format(datetime.now() - start_time)
    print_progress_bar(frameIndex, videoFileFramesTotalLength, "Frames: ", currentTime)

    

    #frameIndex += howManyFramesToIterateBy
    #videoFile.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
    frameIndex += 1
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