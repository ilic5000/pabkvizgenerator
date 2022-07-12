from time import sleep
import cv2
import numpy
import sys
import easyocr

# Configuration ##################################################

fileName = 'examples/slagalica-nova-pitanje-odgovor.png'
#fileName = 'examples/slagalica-stara-pitanje-odgovor.png'

writeDebugInfoOnImages = True
writeDebugInfoOnImagesMaskContours = True
preprocessImageBeforeOCR = True
percentageOfAreaThreshold = 0.6
resizeImagePercentage = 0.5

font = cv2.FONT_HERSHEY_COMPLEX

# OCR language (either latin or cyrillic, cannot do both at the same time)
#ocrLanguage = 'rs_latin'
ocrLanguage = 'rs_cyrillic'

###############################################################################################

def nothing(x):
    # dummy
    pass

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
    return imageToProcess

# Start processing...

# Load model into the memory
reader = easyocr.Reader([ocrLanguage], gpu=False)

cv2.namedWindow("HSVTrackbarsBlue")
cv2.createTrackbar("Lower-H", "HSVTrackbarsBlue", 5, 180, nothing)
cv2.createTrackbar("Lower-S", "HSVTrackbarsBlue", 0, 255, nothing)
cv2.createTrackbar("Lower-V", "HSVTrackbarsBlue", 0, 255, nothing)

cv2.createTrackbar("Upper-H", "HSVTrackbarsBlue", 152, 180, nothing)
cv2.createTrackbar("Upper-S", "HSVTrackbarsBlue", 75, 255, nothing)
cv2.createTrackbar("Upper-V", "HSVTrackbarsBlue", 119, 255, nothing)

image = cv2.imread(fileName)

if image is None:
        print('Failed to load image file:', fileName)
        sys.exit(1)

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

    # ectedTest = reader.detect(answerRectangleImage)
    # if len((detectedTest[0])[0]) > 0:
    #     print("text detected")

    blue_l_h = cv2.getTrackbarPos("Lower-H", "HSVTrackbarsBlue")
    blue_l_s = cv2.getTrackbarPos("Lower-S", "HSVTrackbarsBlue")
    blue_l_v = cv2.getTrackbarPos("Lower-V", "HSVTrackbarsBlue")

    blue_u_h = cv2.getTrackbarPos("Upper-H", "HSVTrackbarsBlue")
    blue_u_s = cv2.getTrackbarPos("Upper-S", "HSVTrackbarsBlue")
    blue_u_v = cv2.getTrackbarPos("Upper-V", "HSVTrackbarsBlue")

    blue_lower_hsv = numpy.array([blue_l_h, blue_l_s, blue_l_v])
    blue_upper_hsv = numpy.array([blue_u_h, blue_u_s, blue_u_v])
    
    blue_mask = cv2.inRange(questionRectangleImage, blue_lower_hsv, blue_upper_hsv)

    # Erode blue mask
    kernelBlue = numpy.ones((3,3), numpy.uint8)
    blue_mask = cv2.erode(blue_mask, kernelBlue)

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
                blue_ymin, blue_ymax, blue_xmin, blue_xmax = calculateMinMaxPoints(font, questionRectangleImage, questionImgHeight, questionImgWidth, approx) 
                maxBlueArea = area
                maxBlueAreaContour = scale_contour(cnt, 1.01)
                maxBlueAreaContourApprox = scale_contour(approx, 1.01)

    if maxBlueArea > 0:
            if writeDebugInfoOnImagesMaskContours:
                print(maxBlueArea)
                cv2.drawContours(questionRectangleImage, [maxBlueAreaContourApprox], 0, (255, 0, 0), 2)

    cv2.imshow("question", questionRectangleImage)
    cv2.imshow("answer", answerRectangleImage)            
    cv2.imshow('blue mask', blue_mask)
    cv2.imshow('Original preview', original_img_preview)

    key = cv2.waitKey(1)
    if key == 27: # ESC
        break

if preprocessImageBeforeOCR:
    questionRectangleImage = preprocessBeforeOCR(questionRectangleImage, invertColors=True)
    answerRectangleImage = preprocessBeforeOCR(answerRectangleImage, invertColors=False)                    

cv2.imwrite("results/%s-question.jpg" %fileName, questionRectangleImage)
ocrQuestionList = reader.readtext(questionRectangleImage, detail = 0, paragraph=True)
ocrQuestion = listToString(ocrQuestionList)

cv2.imwrite("results/%s-answer.jpg" %fileName, answerRectangleImage)
ocrAnswerList = reader.readtext(answerRectangleImage, detail = 0, paragraph=True)
ocrAnswer = listToString(ocrAnswerList)

print('Question: %s' %ocrQuestion)
print('Answer: %s' %ocrAnswer)
print('Success!')


print('Done.')