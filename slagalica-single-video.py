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
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Hardcoded values
templateToFindGameIntro = cv2.imread('examples/slagalica/slagalica-nova-ko-zna-zna-template.png', 0)
templateToFindNextGameIntro = cv2.imread('examples/slagalica/slagalica-nova-asoc-template.png', 0)

diffSimilarirtyAnswerLowerValue = 0.1
diffSimilarirtyAnswerUpperValue = 0.7

# When answer/question are found, jump frames in order to avoid multiple detection of the same question
# This can be done smarter, but this simple jump works just fine
howManyFramesToJumpAfterSuccess = 25
frameIterationStepModifier = 1

# Arguments
parser = argparse.ArgumentParser(description="Slagalica single video processor",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-srcdir", "--srcDirectory", help="directory where file is located", default="examples")
parser.add_argument("-file", "--fileName", help="video file name to be processed", default="Slagalica 01.01.2020. (1080p_25fps_H264-128kbit_AAC).mp4")
parser.add_argument("-o", "--output", help="directory for csv and debug data output", default="results")
parser.add_argument("-lang", "--language", help="ocr language, can be either rs_latin or rs_cyrillic", default="rs_cyrillic")
parser.add_argument("-csv", "--csvFileName", help="name for csv file", default="questions.csv")
parser.add_argument("-d", "--debugData", help="create frame image files for every image processed. note: can use up a lot of data space!", default="True")
parser.add_argument("-poi", "--preprocessOCRImages", help="apply processing (blur, threshold, etc.) before doing ocr to images", default="True")
parser.add_argument("-feocr", "--forceEasyOCR", help="force using of slower EasyOCR instead of default pytesseract", default="False")
args = parser.parse_args()
config = vars(args)

# Configuration ################################################################

srcDir = config['srcDirectory']
fileName = config['fileName']
filePath = "%s/%s" %(srcDir, config['fileName'])
directoryOutput = config['output']
csvFileName = config['csvFileName']
createDebugData = (config['debugData'] == 'True')
preprocessImageBeforeOCR = (config['preprocessOCRImages'] == 'True')
forceUseOfEasyOCR = (config['forceEasyOCR'] == 'True')

# OCR language (either latin or cyrillic, cannot do both at the same time)
ocrLanguage = config['language']

# Found contours area size treshold 
percentageOfAreaThreshold = 0.6

# HSV masks values
blue_l_h = 100
blue_l_s = 118
blue_l_v = 42
blue_u_h = 120
blue_u_s = 255
blue_u_v = 210


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

def listToString(listWords):
    result = " "
    for word in listWords:
        result += word.upper()
    return result

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
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).copy()
    questionImgHeight, questionImgWidth, _ = hsvImage.shape 

    blue_lower_hsv = numpy.array([blue_l_h, blue_l_s, blue_l_v])
    blue_upper_hsv = numpy.array([blue_u_h, blue_u_s, blue_u_v])
    blue_mask = cv2.inRange(hsvImage, blue_lower_hsv, blue_upper_hsv)
    kernelBlue = numpy.ones((3,3), numpy.uint8)
    blue_mask = cv2.erode(blue_mask, kernelBlue)
    contoursInBlueMask, _ = cv2.findContours(blue_mask, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)

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

def preprocessBeforeOCR(imageToProcess, lower_bound, upper_bound, type, useGaussianBlurBefore, useBlurAfter):
    hsv = cv2.cvtColor(imageToProcess, cv2.COLOR_RGB2HSV)
    h, s, v1 = cv2.split(hsv)

    result = v1

    if useGaussianBlurBefore:
        result = cv2.GaussianBlur(v1,(5,5),0)

    # Can be played with... 
    result = cv2.threshold(v1, lower_bound, upper_bound, type)[1]
    
    if useBlurAfter:
        result = cv2.medianBlur(result, 3)

    return result

def easyOCR(reader, image):
    ocrQuestionList = reader.readtext(image, detail = 0, paragraph=True, x_ths = 1000, y_ths = 1000)
    ocrQuestion = listToString(ocrQuestionList)
    return ocrQuestion

def pytesseractOCR(image, handleIncorrectQuestionMarkAtTheEnd):
    recognizedText = pytesseract.image_to_string(image, lang='srp+srp_latn')
    recognizedText = " ".join(recognizedText.split())
    recognizedText = recognizedText.replace('|','')
    recognizedText = recognizedText.replace('\n',' ')
    recognizedText.replace("  ", " ")
    recognizedText = recognizedText.strip()
    recognizedText = " ".join(recognizedText.split())
    recognizedText = recognizedText.upper()

    if handleIncorrectQuestionMarkAtTheEnd:
        recognizedText = recognizedText.rstrip('2')
        recognizedText = recognizedText.rstrip(':2')
        recognizedText = "%s%s" %(recognizedText, '?')
    return recognizedText

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
reader = None
if forceUseOfEasyOCR:
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

    # Show preview of processing... 
    original_img_preview = cv2.resize(originalFrame, (0, 0), fx=0.4, fy=0.4)
    cv2.imshow('Processing video...', original_img_preview)
    key = cv2.waitKey(1)

    # Stats
    currentTime = 'Duration: {}'.format(datetime.now() - start_time)
    print_progress_bar(frameIndex, videoFileFramesTotalLength, "Frames: ", currentTime)

    # MAGIC!
    if not gameFound and does_template_exist(originalFrame, templateToFindGameIntro, confidenceLevel = 0.5):
        gameFound = True
        print("\nGame start found. Frame: %d" %frameIndex)
        gameFoundFrame = originalFrame.copy()
        gameFoundFrame_preview = cv2.resize(originalFrame, (0, 0), fx=0.2, fy=0.2)
        cv2.imshow('Game start:', gameFoundFrame_preview)
        key = cv2.waitKey(1)

    if gameFound: # commonly known as "else"
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
                debugFrameName = "%s/%s-%d-2.1-question.jpg" % (directoryOutput, fileName, frameIndex)
                cv2.imwrite(debugFrameName, questionRectangleImage)
                debugFrameName = "%s/%s-%d-3.1-answer.jpg" % (directoryOutput, fileName, frameIndex)
                cv2.imwrite(debugFrameName, answerRectangleImage)

            if preprocessImageBeforeOCR:
                # OTSU is better for question rectangle - where white text is taking a lot of area and is dominating the image
                # In the answer rectangle however, if answer is really short, OTCU can messup, so there we are using global threshold
                
                questionRectangleImage = preprocessBeforeOCR(questionRectangleImage.copy(), lower_bound=241, upper_bound=255, 
                                                                type=cv2.THRESH_BINARY + cv2.THRESH_OTSU, useGaussianBlurBefore=True, useBlurAfter=True)
                debugFrameName = "%s/%s-%d-2.2-question.jpg" % (directoryOutput, fileName, frameIndex)
                cv2.imwrite(debugFrameName, questionRectangleImage)
                answerRectangleImage = preprocessBeforeOCR(answerRectangleImage.copy(), lower_bound=241, upper_bound=255, 
                                                                type=cv2.THRESH_BINARY, useGaussianBlurBefore=True, useBlurAfter=True)
                debugFrameName = "%s/%s-%d-3.2-answer.jpg" % (directoryOutput, fileName, frameIndex)
                cv2.imwrite(debugFrameName, answerRectangleImage)   

            if forceUseOfEasyOCR:
                ocrQuestion = easyOCR(reader, questionRectangleImage)
                ocrAnswer= easyOCR(reader, answerRectangleImage)
            else: # the default one
                ocrQuestion = pytesseractOCR(questionRectangleImage, handleIncorrectQuestionMarkAtTheEnd = True)
                ocrAnswer= pytesseractOCR(answerRectangleImage, handleIncorrectQuestionMarkAtTheEnd = False)

            print('\n#%d Question: %s' % (numberOfFoundQuestionAnswerPair+1, ocrQuestion))
            print('Answer: %s' %ocrAnswer)

            numberOfFoundQuestionAnswerPair += 1

            with open(csvResultsFileLocation, 'a+', encoding='UTF8', newline='') as f:
                writer = csv.writer(f, delimiter =';')
                csvDataRow = [ocrQuestion, ocrAnswer, bitrate, imageHeight, imageWidth, filePath, frameIndex]
                writer.writerow(csvDataRow)

            frameIndex += howManyFramesToJumpAfterSuccess
            print("\nJumping to %dth frame of %d after found question/answer..." %(frameIndex, videoFileFramesTotalLength))
            if frameIndex >= videoFileFramesTotalLength:
                print("No more frames to process after frame jump...")

        # TRY TO FIND END OF THE GAME
        if(numberOfFoundQuestionAnswerPair == 10):
            print("\nGame end found. 10 Questions reached. Frame: %d" %frameIndex)
            break
        if(does_template_exist(originalFrame, templateToFindNextGameIntro, confidenceLevel = 0.6)):
            
            print("\nQuestions missed: %d" %(10-numberOfFoundQuestionAnswerPair))
            print("Game end found. New game intro recognized. Frame: %d" %frameIndex)
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