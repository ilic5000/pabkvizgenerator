from time import sleep
import cv2
import numpy
import sys

fileName = 'potjera-e1320-frame.jpg'
#fileName = 'potera-srpska.png'
#fileName = 'potera-srpska-2.png'
#fileName = 'potera-srpska-3.png'
#fileName = 'prosta-slika-test.png'

writeDebugInfoOnImages = False
percentageOfAreaThreshold = 0.0030
resizeImagePercentage = 0.5
font = cv2.FONT_HERSHEY_COMPLEX


def nothing(x):
    # dummy
    pass

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


cv2.namedWindow("HSVTrackbarsGreen")
cv2.createTrackbar("Lower-H", "HSVTrackbarsGreen", 31, 180, nothing)
cv2.createTrackbar("Lower-S", "HSVTrackbarsGreen", 23, 255, nothing)
cv2.createTrackbar("Lower-V", "HSVTrackbarsGreen", 0, 255, nothing)

cv2.createTrackbar("Upper-H", "HSVTrackbarsGreen", 84, 180, nothing)
cv2.createTrackbar("Upper-S", "HSVTrackbarsGreen", 255, 255, nothing)
cv2.createTrackbar("Upper-V", "HSVTrackbarsGreen", 255, 255, nothing)

cv2.namedWindow("HSVTrackbarsBlue")
cv2.createTrackbar("Lower-H", "HSVTrackbarsBlue", 100, 180, nothing)
cv2.createTrackbar("Lower-S", "HSVTrackbarsBlue", 118, 255, nothing)
cv2.createTrackbar("Lower-V", "HSVTrackbarsBlue", 42, 255, nothing)

cv2.createTrackbar("Upper-H", "HSVTrackbarsBlue", 120, 180, nothing)
cv2.createTrackbar("Upper-S", "HSVTrackbarsBlue", 255, 255, nothing)
cv2.createTrackbar("Upper-V", "HSVTrackbarsBlue", 210, 255, nothing)

image = cv2.imread(fileName)

if image is None:
        print('Failed to load image file:', fileName)
        sys.exit(1)

hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

while True:
    # must be in the loop (reload empty image/frame)
    original_img_preview = cv2.resize(image, (0, 0), fx=resizeImagePercentage, fy=resizeImagePercentage)
    original_img_previewHeight, original_img_previewWidth, channels = original_img_preview.shape 

    seekAreaBorderHorizontalY = 2 * int(original_img_previewHeight/3)
    seekAreaBorderHorizontalXStart = 0
    seekAreaBorderHorizontalXEnd = original_img_previewWidth

    if writeDebugInfoOnImages:
        cv2.line(original_img_preview, (seekAreaBorderHorizontalXStart, seekAreaBorderHorizontalY), (seekAreaBorderHorizontalXEnd, seekAreaBorderHorizontalY), (0, 255, 0), thickness=2)
    
    seekAreaBorderLeftX = int(original_img_previewWidth/9.1)
    seekAreaBorderLeftY = original_img_previewHeight

    if writeDebugInfoOnImages: 
        cv2.line(original_img_preview, (seekAreaBorderLeftX, seekAreaBorderHorizontalY), (seekAreaBorderLeftX, seekAreaBorderLeftY), (0, 255, 0), thickness=2)

    seekAreaBorderRightX = int(8.1 * int(original_img_previewWidth/9.1))
    seekAreaBorderRightY = original_img_previewHeight

    if writeDebugInfoOnImages:
        cv2.line(original_img_preview, (seekAreaBorderRightX, seekAreaBorderHorizontalY), (seekAreaBorderRightX, seekAreaBorderRightY), (0, 255, 0), thickness=2)

    green_l_h = cv2.getTrackbarPos("Lower-H", "HSVTrackbarsGreen")
    green_l_s = cv2.getTrackbarPos("Lower-S", "HSVTrackbarsGreen")
    green_l_v = cv2.getTrackbarPos("Lower-V", "HSVTrackbarsGreen")

    green_u_h = cv2.getTrackbarPos("Upper-H", "HSVTrackbarsGreen")
    green_u_s = cv2.getTrackbarPos("Upper-S", "HSVTrackbarsGreen")
    green_u_v = cv2.getTrackbarPos("Upper-V", "HSVTrackbarsGreen")

    green_lower_hsv = numpy.array([green_l_h, green_l_s, green_l_v])
    green_upper_hsv = numpy.array([green_u_h, green_u_s, green_u_v])
    
    green_mask = cv2.inRange(hsvImage, green_lower_hsv, green_upper_hsv)

    blue_l_h = cv2.getTrackbarPos("Lower-H", "HSVTrackbarsBlue")
    blue_l_s = cv2.getTrackbarPos("Lower-S", "HSVTrackbarsBlue")
    blue_l_v = cv2.getTrackbarPos("Lower-V", "HSVTrackbarsBlue")

    blue_u_h = cv2.getTrackbarPos("Upper-H", "HSVTrackbarsBlue")
    blue_u_s = cv2.getTrackbarPos("Upper-S", "HSVTrackbarsBlue")
    blue_u_v = cv2.getTrackbarPos("Upper-V", "HSVTrackbarsBlue")

    blue_lower_hsv = numpy.array([blue_l_h, blue_l_s, blue_l_v])
    blue_upper_hsv = numpy.array([blue_u_h, blue_u_s, blue_u_v])
    
    blue_mask = cv2.inRange(hsvImage, blue_lower_hsv, blue_upper_hsv)

    # Erode green mask
    kernelGreen = numpy.ones((5,5), numpy.uint8)
    green_mask = cv2.erode(green_mask, kernelGreen)

    # Erode blue mask
    kernelBlue = numpy.ones((5,5), numpy.uint8)
    blue_mask = cv2.erode(blue_mask, kernelBlue)

    # if image is too big, scale it down for preview
    green_mask_half = cv2.resize(green_mask, (0, 0), fx=resizeImagePercentage, fy=resizeImagePercentage)
    blue_mask_half = cv2.resize(blue_mask, (0, 0), fx=resizeImagePercentage, fy=resizeImagePercentage)

    # Errode green mask half
    kernel_half_green = numpy.ones((2,2), numpy.uint8)
    green_mask_half = cv2.erode(green_mask_half, kernel_half_green)

    # Errode blue mask half
    kernel_half_blue = numpy.ones((2,2), numpy.uint8)
    blue_mask_half = cv2.erode(blue_mask_half, kernel_half_blue)

    contoursInGreenMask, _ = cv2.findContours(green_mask_half, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
    contoursInBlueMask, _ = cv2.findContours(blue_mask_half, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)

    totalPixels = original_img_previewHeight * original_img_previewWidth
    areaThreashold = percentageOfAreaThreshold * totalPixels

    maxGreenArea = 0 
    maxGreenAreaContour = None
    maxGreenAreaContourApprox = None
    green_ymin, green_ymax, green_xmin, green_xmax = None, None, None, None

    for cnt in contoursInGreenMask:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        numberOfPoints = len(approx)

        if area > maxGreenArea and numberOfPoints >= 4 and numberOfPoints <= 6 and area > areaThreashold:
            green_ymin, green_ymax, green_xmin, green_xmax = calculateMinMaxPoints(font, original_img_preview, original_img_previewHeight, original_img_previewWidth, approx)
            if green_ymin > seekAreaBorderHorizontalY and green_xmin > seekAreaBorderHorizontalXStart and green_xmax < seekAreaBorderHorizontalXEnd:
                maxGreenArea = area
                maxGreenAreaContour = scale_contour(cnt, 1.01)
                maxGreenAreaContourApprox = scale_contour(approx, 1.01)
    
    if maxGreenArea > 0:
        if writeDebugInfoOnImages:
            cv2.drawContours(original_img_preview, [maxGreenAreaContourApprox], 0, (0, 255, 0), 2)

    maxBlueArea = 0 
    maxBlueAreaContour = None
    maxBlueAreaContourApprox = None
    blue_ymin, blue_ymax, blue_xmin, blue_xmax = None, None, None, None

    for cnt in contoursInBlueMask:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        numberOfPoints = len(approx)
        
        if area > maxBlueArea and numberOfPoints >= 4 and numberOfPoints <= 6 and area > areaThreashold and area > 3 * maxGreenArea :
            blue_ymin, blue_ymax, blue_xmin, blue_xmax = calculateMinMaxPoints(font, original_img_preview, original_img_previewHeight, original_img_previewWidth, approx) 
            if blue_ymin > seekAreaBorderHorizontalY and blue_xmin > seekAreaBorderHorizontalXStart and blue_xmax < seekAreaBorderHorizontalXEnd:   
                maxBlueArea = area
                maxBlueAreaContour = scale_contour(cnt, 1.01)
                maxBlueAreaContourApprox = scale_contour(approx, 1.01)

    if maxBlueArea > 0:
            if writeDebugInfoOnImages:
                cv2.drawContours(original_img_preview, [maxBlueAreaContourApprox], 0, (255, 0, 0), 2)
            
    cv2.imshow('green mask', green_mask_half)
    cv2.imshow('blue mask', blue_mask_half)
    cv2.imshow('Original preview', original_img_preview)

    key = cv2.waitKey(1)
    if key == 27: # ESC
        break

if maxGreenArea > 0 and maxBlueArea > 0:
    questionRectangleImage = original_img_preview[blue_ymin:blue_ymax, blue_xmin:blue_xmax]
    answerRectangleImage = original_img_preview[green_ymin:green_ymax, green_xmin:green_xmax]
    cv2.imwrite("results/question.jpg", questionRectangleImage) 
    cv2.imwrite("results/answer.jpg", answerRectangleImage)
    print('Success!')
else:
    print('Error: Question/Answer not found!')

print('Done.')