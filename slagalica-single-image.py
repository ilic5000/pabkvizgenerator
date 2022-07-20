from time import sleep
import cv2
import numpy
import sys
import easyocr
import pytesseract
from skimage.morphology import skeletonize
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configuration ##################################################

fileDir = 'examples'
fileName = '720p-intro1-mask-ready.png'
fileName2 = '720p-intro2-mask-ready.png'

filePath = "%s/%s"%(fileDir,fileName)
filePath2 = "%s/%s"%(fileDir,fileName2)

# Use tesseract by default
forceEasyOCR = False

writeDebugInfoOnImages = True
writeDebugInfoOnImagesMaskContours = True
preprocessQuestionImageBeforeOCR = False
preprocessAnswerImageBeforeOCR = False

percentageOfAreaThreshold = 0.6
resizeImagePercentage = 0.5

font = cv2.FONT_HERSHEY_COMPLEX

# Easy OCR language (either latin or cyrillic, cannot do both at the same time), 'en' is always added alongside one of these
#ocrLanguage = 'rs_latin'
easyOcrLanguage = 'rs_cyrillic'

# pytesseract
pytesseractLang = 'srp+srp_latn'

###############################################################################################

def nothing(x):
    # dummy
    pass

def listToString(listWords):
    result = ""
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

def preprocessBeforeOCROld(imageToProcess, invertColors):
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

def removeNewlines(value):
    return value.replace('\n',' ')

def easyOCR(reader, image):
    ocrQuestionList = reader.readtext(image, detail = 0, paragraph=True, x_ths = 1000, y_ths = 1000)
    ocrQuestion = listToString(ocrQuestionList)
    return ocrQuestion

def pytesseractOCR(image):
    recognizedText = pytesseract.image_to_string(image, lang=pytesseractLang)
    recognizedText = removeNewlines(recognizedText)

    return recognizedText

######################################################################################
# Start processing...

# Load model into the memory
reader = None
if forceEasyOCR:
    reader = easyocr.Reader(['en', easyOcrLanguage], gpu=False)

cv2.namedWindow("HSVTrackbarsBlue")
cv2.createTrackbar("Lower-H", "HSVTrackbarsBlue", 100, 180, nothing)
cv2.createTrackbar("Lower-S", "HSVTrackbarsBlue", 118, 255, nothing)
cv2.createTrackbar("Lower-V", "HSVTrackbarsBlue", 42, 255, nothing)

cv2.createTrackbar("Upper-H", "HSVTrackbarsBlue", 122, 180, nothing)
cv2.createTrackbar("Upper-S", "HSVTrackbarsBlue", 255, 255, nothing)
cv2.createTrackbar("Upper-V", "HSVTrackbarsBlue", 210, 255, nothing)

image = cv2.imread(filePath)
image2 = cv2.imread(filePath2)

if image is None:
        print('Failed to load image file:', filePath)
        sys.exit(1)

image = unsharp_mask(image)

hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

while True:
    # must be in the loop (reload empty image/frame)
    original_img_preview = cv2.resize(image, (0, 0), fx=resizeImagePercentage, fy=resizeImagePercentage)
    original_img_preview2 = cv2.resize(image2, (0, 0), fx=resizeImagePercentage, fy=resizeImagePercentage)

    original_img_previewHeight, original_img_previewWidth, channels = original_img_preview.shape 

    seekAreaQuestionBorderUpperLineY = int(5.95 * int(original_img_previewHeight/10))
    seekAreaQuestionBorderLowerLineY = int(8.22 * int(original_img_previewHeight/10))
    seekAreaAnswerBorderLowerLineY = int(9.0 * int(original_img_previewHeight/10))

    seekAreaBorderLeftX = int(1.13 * original_img_previewWidth/10)
    seekAreaBorderLeftY = seekAreaAnswerBorderLowerLineY

    seekAreaBorderRightX = int(8.1 * int(original_img_previewWidth/9.1))
    seekAreaBorderRightY = seekAreaAnswerBorderLowerLineY

    if writeDebugInfoOnImages:
        cv2.line(original_img_preview, (seekAreaBorderLeftX, seekAreaQuestionBorderUpperLineY), (seekAreaBorderRightX, seekAreaQuestionBorderUpperLineY), (0, 255, 0), thickness=1)
        cv2.line(original_img_preview, (seekAreaBorderLeftX, seekAreaQuestionBorderLowerLineY), (seekAreaBorderRightX, seekAreaQuestionBorderLowerLineY), (0, 255, 255), thickness=1)
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

    original_img_previewhsv = cv2.cvtColor(original_img_preview.copy(), cv2.COLOR_BGR2HSV)
    original_img_preview2hsv = cv2.cvtColor(original_img_preview2.copy(), cv2.COLOR_BGR2HSV)
    firstimagemask = cv2.inRange(original_img_previewhsv, blue_lower_hsv, blue_upper_hsv)
    secondimagemask = cv2.inRange(original_img_preview2hsv, blue_lower_hsv, blue_upper_hsv)

    cv2.imshow("firstimage", original_img_preview)
    cv2.imshow("secondimage", original_img_preview2)
    cv2.imshow("firstimage mask", firstimagemask)
    cv2.imshow("secondimage mask", secondimagemask)

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

    if maxBlueArea > 0:
        #Contour found
            if writeDebugInfoOnImagesMaskContours:
                #print(maxBlueArea)
                cv2.drawContours(questionRectangleImage, [maxBlueAreaContourApprox], 0, (255, 0, 0), 2)
                
                temp = cv2.cvtColor(blue_mask.copy(),cv2.COLOR_GRAY2RGB)
                cv2.drawContours(temp, [maxBlueAreaContourApprox], 0, (0, 0, 255), 2)
                cv2.imshow('test', temp)
                cv2.imwrite('results\yyy.jpg', temp)
                key = cv2.waitKey(1)

    cv2.imshow("question", questionRectangleImage)
    cv2.imshow("answer", answerRectangleImage)            
    cv2.imshow('blue mask', blue_mask)
    cv2.imshow('Original preview', original_img_preview)

    key = cv2.waitKey(1)
    if key == 27: # ESC
        break

if preprocessQuestionImageBeforeOCR:
    questionRectangleImage = preprocessBeforeOCR(questionRectangleImage.copy(), lower_bound=222, upper_bound=255, 
                                                    type=cv2.THRESH_BINARY, useGaussianBlurBefore=True, useBlurAfter=True)
if preprocessAnswerImageBeforeOCR:
    answerRectangleImage = preprocessBeforeOCR(answerRectangleImage.copy(), lower_bound=241, upper_bound=255, 
                                                    type=cv2.THRESH_BINARY, useGaussianBlurBefore=True, useBlurAfter=True)                    

cv2.imwrite("results/%s-question.jpg" %fileName, questionRectangleImage)
cv2.imwrite("results/%s-answer.jpg" %fileName, answerRectangleImage)

if forceEasyOCR:
    print('EasyOCR: ')
    ocrQuestion = easyOCR(reader, questionRectangleImage)
    print(ocrQuestion)
    ocrAnswer = easyOCR(reader, answerRectangleImage)
    print(ocrAnswer)

print('pytesseractOCR: ')
ocrQuestion = pytesseractOCR(questionRectangleImage)
ocrAnswer = pytesseractOCR(answerRectangleImage)

print('Question: %s' %ocrQuestion)
print('Answer: %s' %ocrAnswer)
print('Success!')

print('Done.')