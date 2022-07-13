from time import sleep
import cv2
import numpy
import sys
import easyocr
from skimage.morphology import skeletonize

# Configuration ##################################################

fileDir = 'examples'
fileName = 'Slagalica 01.01.2020. (1080p_25fps_H264-128kbit_AAC).mp4-23267-0-frame.jpg'
filePath = "%s/%s"%(fileDir,fileName)

writeDebugInfoOnImages = True
writeDebugInfoOnImagesMaskContours = True
preprocessQuestionImageBeforeOCR = True
preprocessAnswerImageBeforeOCR = True

percentageOfAreaThreshold = 0.6
resizeImagePercentage = 1

font = cv2.FONT_HERSHEY_COMPLEX

# OCR language (either latin or cyrillic, cannot do both at the same time)
#ocrLanguage = 'rs_latin'
ocrLanguage = 'rs_cyrillic'

###############################################################################################

def nothing(x):
    # dummy
    pass

def listToString(listWords):
    result = " "
    for word in listWords:
        result += word.upper()
    return result

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

def areAllPointsInsideSeekBorderArea(approx, seekAreaBorderHorizontalY, seekAreaBorderHorizontalXStart, seekAreaBorderHorizontalXEnd):
    result = True 
    n = approx.ravel() 
    i = 0
    for j in n :
        if(i % 2 == 0):
            x = n[i]
            y = n[i + 1]
            if y < seekAreaBorderHorizontalY or x < seekAreaBorderHorizontalXStart or x > seekAreaBorderHorizontalXEnd:
                result = False
                break
        
        i = i + 1

    return result

def calculateMinMaxPoints(font, original_img_preview, original_img_previewHeight, original_img_previewWidth, approx):
    #https://www.geeksforgeeks.org/find-co-ordinates-of-contours-using-opencv-python/
    n = approx.ravel() 
    i = 0
    ymin = original_img_previewHeight
    ymax = 0
    xmin = original_img_previewWidth
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
            
            if writeDebugInfoOnImages:
                textCoordinate = str(x) + " " + str(y) 
                cv2.putText(original_img_preview, textCoordinate, (x, y), 
                                        font, 0.5, (0, 0, 255)) 
        i = i + 1
    return ymin,ymax,xmin,xmax

def preprocessBeforeOCR(imageToProcess, invertColors):
    imageToProcess = cv2.cvtColor(imageToProcess, cv2.COLOR_BGR2GRAY)
    if invertColors:
        imageToProcess = cv2.bitwise_not(imageToProcess)
    dilated_img = cv2.dilate(imageToProcess, numpy.ones((7, 7), numpy.uint8))
    bg_img = cv2.medianBlur(dilated_img, 17)
    diff_img = 255 - cv2.absdiff(imageToProcess, bg_img)
    imageToProcess = cv2.normalize(diff_img, None, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    imageToProcess = cv2.threshold(imageToProcess, 0, 255, cv2.THRESH_OTSU)[1]

    cv2.imshow('preprocessBeforeOCRTest 1', imageToProcess)
    key = cv2.waitKey()
    return imageToProcess

def preprocessBeforeOCRTest(imageToProcess, invertColors):
    hsv = cv2.cvtColor(imageToProcess, cv2.COLOR_RGB2HSV)
    h, s, v1 = cv2.split(hsv)

    #test1a = imageToProcess.astype("CV_16UC1")
    threasholdApplied = cv2.threshold(v1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imshow('Threashold applied', threasholdApplied)

    medianBlur = cv2.medianBlur(threasholdApplied, 3)
    cv2.imshow('Median blur applied', medianBlur)
    key = cv2.waitKey()
    
    # # Define range of white color in HSV
    # lower_white = numpy.array([0, 0, 184])
    # upper_white = numpy.array([178, 239, 255])
    # # Threshold the HSV image
    # maskWhite = cv2.inRange(hsv, lower_white, upper_white)

    # # Remove noise
    # kernel_erode = numpy.ones((2,2), numpy.uint8)
    # eroded_mask = cv2.erode(maskWhite, kernel_erode, iterations=2)

    # kernel_dilate = numpy.ones((3,3),numpy.uint8)
    # dilated_mask = cv2.dilate(maskWhite, kernel_dilate, iterations=1)
    
    # medianBlur = cv2.medianBlur(eroded_mask, 5)
    # medianBlur2 = cv2.medianBlur(maskWhite, 5)

    # cv2.imshow('preprocessBeforeOCRTest 1', maskWhite)
    # key = cv2.waitKey()
    # cv2.imshow('preprocessBeforeOCRTest 2', eroded_mask)
    # key = cv2.waitKey()
    # cv2.imshow('preprocessBeforeOCRTest 3', medianBlur)
    # key = cv2.waitKey()
    # cv2.imshow('preprocessBeforeOCRTest 4', medianBlur2)
    # key = cv2.waitKey()

    return medianBlur

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = numpy.maximum(sharpened, numpy.zeros(sharpened.shape))
    sharpened = numpy.minimum(sharpened, 255 * numpy.ones(sharpened.shape))
    sharpened = sharpened.round().astype(numpy.uint8)
    if threshold > 0:
        low_contrast_mask = numpy.absolute(image - blurred) < threshold
        numpy.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
# Start processing...

# Load model into the memory
reader = easyocr.Reader(['en', ocrLanguage], gpu=False)

cv2.namedWindow("HSVTrackbarsBlue")
cv2.createTrackbar("Lower-H", "HSVTrackbarsBlue", 100, 180, nothing)
cv2.createTrackbar("Lower-S", "HSVTrackbarsBlue", 118, 255, nothing)
cv2.createTrackbar("Lower-V", "HSVTrackbarsBlue", 42, 255, nothing)

cv2.createTrackbar("Upper-H", "HSVTrackbarsBlue", 120, 180, nothing)
cv2.createTrackbar("Upper-S", "HSVTrackbarsBlue", 255, 255, nothing)
cv2.createTrackbar("Upper-V", "HSVTrackbarsBlue", 210, 255, nothing)

image = cv2.imread(filePath)

if image is None:
        print('Failed to load image file:', filePath)
        sys.exit(1)

image = unsharp_mask(image)

hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

while True:
    
    # must be in the loop (reload empty image/frame)
    original_img_preview = cv2.resize(image, (0, 0), fx=resizeImagePercentage, fy=resizeImagePercentage)
    original_img_previewHeight, original_img_previewWidth, channels = original_img_preview.shape 

    seekAreaQuestionBorderUpperLineY = int(5.85 * int(original_img_previewHeight/10))
    seekAreaQuestionBorderLowerLineY = int(8.25 * int(original_img_previewHeight/10))
    seekAreaAnswerBorderLowerLineY = int(9.1 * int(original_img_previewHeight/10))

    seekAreaBorderLeftX = int(original_img_previewWidth/10)
    seekAreaBorderLeftY = seekAreaAnswerBorderLowerLineY

    seekAreaBorderRightX = int(8.2 * int(original_img_previewWidth/9.1))
    seekAreaBorderRightY = seekAreaAnswerBorderLowerLineY

    if writeDebugInfoOnImages:
        cv2.line(original_img_preview, (seekAreaBorderLeftX, seekAreaQuestionBorderUpperLineY), (seekAreaBorderRightX, seekAreaQuestionBorderUpperLineY), (0, 255, 0), thickness=1)
        cv2.line(original_img_preview, (seekAreaBorderLeftX, seekAreaQuestionBorderLowerLineY), (seekAreaBorderRightX, seekAreaQuestionBorderLowerLineY), (0, 255, 255), thickness=2)
        cv2.line(original_img_preview, (seekAreaBorderLeftX, seekAreaAnswerBorderLowerLineY), (seekAreaBorderRightX, seekAreaAnswerBorderLowerLineY), (0, 255, 0), thickness=1)
        
        cv2.line(original_img_preview, (seekAreaBorderLeftX, seekAreaQuestionBorderUpperLineY), (seekAreaBorderLeftX, seekAreaBorderLeftY), (0, 255, 0), thickness=1)
        cv2.line(original_img_preview, (seekAreaBorderRightX, seekAreaQuestionBorderUpperLineY), (seekAreaBorderRightX, seekAreaBorderRightY), (0, 255, 0), thickness=1)

    questionRectangleImage = original_img_preview[seekAreaQuestionBorderUpperLineY:seekAreaQuestionBorderLowerLineY, seekAreaBorderLeftX:seekAreaBorderRightX].copy()
    answerRectangleImage = original_img_preview[seekAreaQuestionBorderLowerLineY:seekAreaAnswerBorderLowerLineY, seekAreaBorderLeftX:seekAreaBorderRightX].copy()

    questionRectangleImageHsvImage = cv2.cvtColor(questionRectangleImage, cv2.COLOR_BGR2HSV)

    blue_l_h = cv2.getTrackbarPos("Lower-H", "HSVTrackbarsBlue")
    blue_l_s = cv2.getTrackbarPos("Lower-S", "HSVTrackbarsBlue")
    blue_l_v = cv2.getTrackbarPos("Lower-V", "HSVTrackbarsBlue")

    blue_u_h = cv2.getTrackbarPos("Upper-H", "HSVTrackbarsBlue")
    blue_u_s = cv2.getTrackbarPos("Upper-S", "HSVTrackbarsBlue")
    blue_u_v = cv2.getTrackbarPos("Upper-V", "HSVTrackbarsBlue")

    blue_lower_hsv = numpy.array([blue_l_h, blue_l_s, blue_l_v])
    blue_upper_hsv = numpy.array([blue_u_h, blue_u_s, blue_u_v])
    
    blue_mask = cv2.inRange(questionRectangleImageHsvImage, blue_lower_hsv, blue_upper_hsv)

    # Erode blue mask
    #kernelBlue = numpy.ones((3,3), numpy.uint8)
    #blue_mask = cv2.erode(blue_mask, kernelBlue)

    contoursInBlueMask, _ = cv2.findContours(blue_mask, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)

    questionImgHeight, questionImgWidth, _ = questionRectangleImage.shape 

    totalPixelsQuestionRectangle = questionImgHeight * questionImgWidth
    areaThreashold = percentageOfAreaThreshold * totalPixelsQuestionRectangle

    maxBlueArea = 0 
    maxBlueAreaContour = None
    maxBlueAreaContourApprox = None
    blue_ymin, blue_ymax, blue_xmin, blue_xmax = None, None, None, None

    for cnt in contoursInBlueMask:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        numberOfPoints = len(approx)
        
        if area > maxBlueArea and numberOfPoints >= 4 and numberOfPoints <= 8 and area > areaThreashold:
                maxBlueArea = area
                maxBlueAreaContour = scale_contour(cnt, 1.01)
                maxBlueAreaContourApprox = scale_contour(approx, 1.01)

    #if maxBlueArea > 0:
        # Contour found
            #if writeDebugInfoOnImagesMaskContours:
                #print(maxBlueArea)
                #cv2.drawContours(questionRectangleImage, [maxBlueAreaContourApprox], 0, (255, 0, 0), 2)

    cv2.imshow("question", questionRectangleImage)
    cv2.imshow("answer", answerRectangleImage)            
    cv2.imshow('blue mask', blue_mask)
    cv2.imshow('Original preview', original_img_preview)

    key = cv2.waitKey(1)
    if key == 27: # ESC
        break

if preprocessQuestionImageBeforeOCR:
    questionRectangleImage = preprocessBeforeOCRTest(questionRectangleImage, invertColors=True)
if preprocessAnswerImageBeforeOCR:
    answerRectangleImage = preprocessBeforeOCRTest(answerRectangleImage, invertColors=False)                    

cv2.imwrite("results/%s-question.jpg" %fileName, questionRectangleImage)
ocrQuestionList = reader.readtext(questionRectangleImage, detail = 0, paragraph=True, x_ths = 1000, y_ths = 1000)
ocrQuestion = listToString(ocrQuestionList)

cv2.imwrite("results/%s-answer.jpg" %fileName, answerRectangleImage)
ocrAnswerList = reader.readtext(answerRectangleImage, batch_size=3, detail = 0, paragraph=True, x_ths = 1000, y_ths = 1000)
ocrAnswer = listToString(ocrAnswerList)

print('Question: %s' %ocrQuestion)
print('Answer: %s' %ocrAnswer)
print('Success!')

print('Done.')