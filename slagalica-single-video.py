from operator import truediv
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
parser.add_argument("-file", "--fileName", help="video file name to be processed", default="Slagalica 01.01.2020. (1080p_25fps_H264-128kbit_AAC).mp4")
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

templateToFindGameIntro = cv2.imread('examples/slagalica-nova-pocetak-template.png', 0)
templateToFindNextGameIntro = cv2.imread('examples/slagalica-nova-asoc-template.png', 0)
writeDebugInfoOnImages = True

diffSimilarirtyAnswerLowerValue = 0.1
diffSimilarirtyAnswerUpperValue = 0.7

# OCR language (either latin or cyrillic, cannot do both at the same time)
ocrLanguage = config['language']

# Found contours area size treshold 
percentageOfAreaThreshold = 0.6

# HSV masks values
blue_l_h = 5
blue_l_s = 0
blue_l_v = 0
blue_u_h = 152
blue_u_s = 75
blue_u_v = 119

# When answer/question are found, jump frames in order to avoid multiple detection of the same question
# This can be done smarter, but this simple jump works just fine
howManyFramesToJumpAfterSuccess = 25
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

def process_img_demo_purposes(img_rgb, template, count):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    templateWidth, templateHeight = template.shape[::-1]
                                                                                                                                                                       
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    cv2.putText(img_rgb, "%s" % max_val, (100,100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    cv2.rectangle(img_rgb, max_loc,  (max_loc[0]+templateWidth , max_loc[1] + templateHeight), (0,255,255), 2)

    cv2.imshow('original', img_rgb)
    key = cv2.waitKey(1)

    if max_val > 0.5:
        cv2.waitKey()
    #cv2.waitKey()
    #cv2.destroyAllWindows()

def does_template_exist(sourceImage, templateToFind, confidenceLevel):
    img_gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)                                                                                                                
    res = cv2.matchTemplate(img_gray, templateToFind, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val >= confidenceLevel:
        return True
    return False

def compare_two_images(sourceImage, templateToFind):
    #img_gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)                                                                                                                
    res = cv2.matchTemplate(sourceImage, templateToFind, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val

def isQuestionsFrameVisible(percentageOfAreaThreshold, blue_l_h, blue_l_s, blue_l_v, blue_u_h, blue_u_s, blue_u_v, image):
    blue_lower_hsv = numpy.array([blue_l_h, blue_l_s, blue_l_v])
    blue_upper_hsv = numpy.array([blue_u_h, blue_u_s, blue_u_v])
    blue_mask = cv2.inRange(image, blue_lower_hsv, blue_upper_hsv)
    kernelBlue = numpy.ones((3,3), numpy.uint8)
    blue_mask = cv2.erode(blue_mask, kernelBlue)
    contoursInBlueMask, _ = cv2.findContours(blue_mask, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
    questionImgHeight, questionImgWidth, _ = image.shape 
    totalPixelsQuestionRectangle = questionImgHeight * questionImgWidth
    areaThreashold = percentageOfAreaThreshold * totalPixelsQuestionRectangle
    maxBlueArea = 0 

    for cnt in contoursInBlueMask:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        numberOfPoints = len(approx)
            
        if area > maxBlueArea and numberOfPoints >= 4 and numberOfPoints <= 8 and area > areaThreashold:
                maxBlueArea = area

    if maxBlueArea > 0:
        return True
    
    return False

def isTextPresentInBothImages(reader, questionRectangleImage, answerRectangleImage):
    # Good, but really slow and requires CUDA cores
    # So, I use it if you have resources
    detectedTestQuestion = reader.detect(questionRectangleImage)
    detectedTestQuestionValue = len((detectedTestQuestion[0])[0])
    detectedTestAnswer = reader.detect(answerRectangleImage)
    detectedTestAnswerValue = len((detectedTestAnswer[0])[0])
    if detectedTestQuestionValue > 0 and detectedTestAnswerValue > 0:
        return True

    return False

def preprocessBeforeOCRTest(imageToProcess, invertColors):
    hsv = cv2.cvtColor(imageToProcess, cv2.COLOR_RGB2HSV)
    # Define range of white color in HSV
    lower_white = numpy.array([0, 0, 184])
    upper_white = numpy.array([178, 239, 255])
    # Threshold the HSV image
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Remove noise
    kernel_erode = numpy.ones((2,2), numpy.uint8)
    eroded_mask = cv2.erode(mask, kernel_erode, iterations=2)
    kernel_dilate = numpy.ones((3,3),numpy.uint8)
    dilated_mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    
    # blur threshold image
    blur = cv2.medianBlur(mask, 3)

    cv2.imshow('preprocessBeforeOCRTest', mask)
    key = cv2.waitKey()
    cv2.imshow('preprocessBeforeOCRTest', blur)
    key = cv2.waitKey()

    return blur

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
reader = easyocr.Reader(['en', ocrLanguage], gpu=True)

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
videoFileFramesTotalLength = int(videoFile.get(cv2.CAP_PROP_FRAME_COUNT))
frameIndex = int(videoFileFramesTotalLength/2) + 3800 #todo remove later 3800
videoFile.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
success,originalFrame = videoFile.read()

# Create seek area (a lot easier to find shapes and avoid false detections on unimportant parts of the image)
imageHeight, imageWidth, _ = originalFrame.shape 

seekAreaQuestionBorderUpperLineY = int(5.85 * int(imageHeight/10))
seekAreaQuestionBorderLowerLineY = int(8.25 * int(imageHeight/10))
seekAreaAnswerBorderLowerLineY = int(9.1 * int(imageHeight/10))

seekAreaBorderLeftX = int(imageWidth/10)
seekAreaBorderLeftY = seekAreaAnswerBorderLowerLineY

seekAreaBorderRightX = int(8.2 * int(imageWidth/9.1))
seekAreaBorderRightY = seekAreaAnswerBorderLowerLineY

# Calculate area of found shapes tresholds
totalPixels = imageHeight * imageWidth
areaThreashold = percentageOfAreaThreshold * totalPixels

skipFirstGreenFoundMaskFrames = True

# Get video bitrate for debug purposes
bitrate = get_bitrate(filePath)

videoAverageFps = get_fps(filePath)
print("FPS: %d" %videoAverageFps)

howManyFramesToIterateBy = int(frameIterationStepModifier * videoAverageFps)
print("Frame iteration step: %d" %howManyFramesToIterateBy)

numberOfFoundQuestionAnswerPair = 0

gameFound = False
questionWithAnswerFrameFound = False
answerRectangleTemp = None
answerRectangleDiffCounter = 0

# Loop through all frames of the video
while success:
    original_img_preview = cv2.resize(originalFrame, (0, 0), fx=0.4, fy=0.4)
    cv2.imshow('Processing video...', original_img_preview)
    key = cv2.waitKey(1)

    currentTime = 'Duration: {}'.format(datetime.now() - start_time)
    print_progress_bar(frameIndex, videoFileFramesTotalLength, "Frames: ", currentTime)

    if not gameFound and does_template_exist(originalFrame, templateToFindGameIntro, confidenceLevel = 0.5):
        gameFound = True
        print("\nGame start found. Frame: %d" %frameIndex)
        gameFoundFrame = originalFrame.copy()
        gameFoundFrame_preview = cv2.resize(originalFrame, (0, 0), fx=0.2, fy=0.2)
        cv2.imshow('Game start:', gameFoundFrame_preview)
        key = cv2.waitKey(1)

    if gameFound: #commonly known as "else"
        questionRectangleImage = originalFrame[seekAreaQuestionBorderUpperLineY:seekAreaQuestionBorderLowerLineY, seekAreaBorderLeftX:seekAreaBorderRightX].copy()
        answerRectangleImage = originalFrame[seekAreaQuestionBorderLowerLineY:seekAreaAnswerBorderLowerLineY, seekAreaBorderLeftX:seekAreaBorderRightX].copy()

        questionFrameVisible = isQuestionsFrameVisible(percentageOfAreaThreshold, blue_l_h, blue_l_s, blue_l_v, blue_u_h, blue_u_s, blue_u_v, questionRectangleImage)

        if questionFrameVisible: 
            if answerRectangleTemp is not None:
                diffSimilarityValue = compare_two_images(answerRectangleImage, answerRectangleTemp)
                if diffSimilarityValue > diffSimilarirtyAnswerLowerValue and diffSimilarityValue < diffSimilarirtyAnswerUpperValue:
                    answerRectangleImage_preview = cv2.resize(answerRectangleImage, (0, 0), fx=0.2, fy=0.2)
                    cv2.imshow('Change detected found:', answerRectangleImage_preview)
                    key = cv2.waitKey(1)
                    print("\ndiffSimilarityValue: {:.2f}".format(diffSimilarityValue))
                    print("answerRectangleDiffCounter: %d" %(answerRectangleDiffCounter))
                    answerRectangleDiffCounter += 1
            answerRectangleTemp = answerRectangleImage.copy()

        questionWithAnswerFrameFound = (answerRectangleDiffCounter % 2 == 1)

        if questionWithAnswerFrameFound:
            if createDebugData:
                cv2.imwrite("results/%s-%d-0-frame.jpg" % (fileName, frameIndex), originalFrame)
                debugCopy = originalFrame.copy()
                cv2.line(debugCopy, (seekAreaBorderLeftX, seekAreaQuestionBorderUpperLineY), (seekAreaBorderRightX, seekAreaQuestionBorderUpperLineY), (0, 255, 0), thickness=1)
                cv2.line(debugCopy, (seekAreaBorderLeftX, seekAreaQuestionBorderLowerLineY), (seekAreaBorderRightX, seekAreaQuestionBorderLowerLineY), (0, 255, 255), thickness=2)
                cv2.line(debugCopy, (seekAreaBorderLeftX, seekAreaAnswerBorderLowerLineY), (seekAreaBorderRightX, seekAreaAnswerBorderLowerLineY), (0, 255, 0), thickness=1)
                cv2.line(debugCopy, (seekAreaBorderLeftX, seekAreaQuestionBorderUpperLineY), (seekAreaBorderLeftX, seekAreaBorderLeftY), (0, 255, 0), thickness=1)
                cv2.line(debugCopy, (seekAreaBorderRightX, seekAreaQuestionBorderUpperLineY), (seekAreaBorderRightX, seekAreaBorderRightY), (0, 255, 0), thickness=1)
                debugFrameName = "%s/%s-%d-1-frame-contours.jpg" % (directoryOutput, fileName, frameIndex)
                cv2.imwrite(debugFrameName, debugCopy)
                debugFrameName = "%s/%s-%d-2-question.jpg" % (directoryOutput, fileName, frameIndex)
                cv2.imwrite(debugFrameName, questionRectangleImage)
                debugFrameName = "%s/%s-%d-3-answer.jpg" % (directoryOutput, fileName, frameIndex)
                cv2.imwrite(debugFrameName, answerRectangleImage)

            if preprocessImageBeforeOCR:
                questionRectangleImage = preprocessBeforeOCR(questionRectangleImage, invertColors=True)
                answerRectangleImage = preprocessBeforeOCR(answerRectangleImage, invertColors=False)   

            ocrQuestionList = reader.readtext(questionRectangleImage, detail = 0, paragraph=True)
            ocrQuestion = listToString(ocrQuestionList)
            ocrAnswerList = reader.readtext(answerRectangleImage, detail = 0, paragraph=True)
            ocrAnswer = listToString(ocrAnswerList)
            print('\n#%d' % (numberOfFoundQuestionAnswerPair+1))
            print('Question: %s' %ocrQuestion)
            print('Answer: %s' %ocrAnswer)

            numberOfFoundQuestionAnswerPair += 1

            with open(csvResultsFileLocation, 'a+', encoding='UTF8', newline='') as f:
                writer = csv.writer(f, delimiter =';')
                csvDataRow = [ocrQuestion, ocrAnswer, bitrate, imageHeight, imageWidth, filePath, frameIndex]
                writer.writerow(csvDataRow)

            frameIndex += howManyFramesToJumpAfterSuccess
            print("\nJump to %dth frame of %d" %(frameIndex, videoFileFramesTotalLength))
            if frameIndex >= videoFileFramesTotalLength:
                print("No more frames to process after frame jump...")

        # TRY TO FIND END OF THE GAME
        if(numberOfFoundQuestionAnswerPair == 10 or does_template_exist(originalFrame, templateToFindNextGameIntro, confidenceLevel = 0.6)):
            # Game finished
            print("\nGame end found. Frame: %d" %frameIndex)
            #cv2.imshow('main window', originalFrame)
            break
        
    #process_img_demo_purposes(originalFrame, templateToFind, frameIndex)
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