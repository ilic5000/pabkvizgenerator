from time import sleep
import cv2
import numpy
import sys

fileName = 'potjera-e1320-frame.jpg'
#fileName = 'potera-srpska.png'
#fileName = 'potera-srpska-2.png'
#fileName = 'prosta-slika-test.png'

percentageOfAreaThreshold = 0.0025
resizeImagePercentage = 0.5
font = cv2.FONT_HERSHEY_COMPLEX

def nothing(x):
    # dummy
    pass

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

    #contours, _ = cv2.findContours(mask, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
    contoursInGreenMask, _ = cv2.findContours(green_mask_half, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
    contoursInBlueMask, _ = cv2.findContours(blue_mask_half, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)

    #for cnt in contours:
        #approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        #cv2.drawContours(image, [approx], 0, (255, 0, 0), 2)

    totalPixels = original_img_previewHeight * original_img_previewWidth
    areaThreashold = percentageOfAreaThreshold * totalPixels
    greenContureArea = 0

    for cnt in contoursInGreenMask:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(original_img_preview, [approx], 0, (0, 0, 200), 1)

        #print('len contours2 %d' %len(contours2)) 
        if len(approx) == 4:
            #print('len approx %d' %len(approx))
            if area > areaThreashold:
                greenContureArea = area
                print('area %d' %area)
                cv2.drawContours(original_img_preview, [approx], 0, (0, 0, 255), 2)

                ymin, ymax, xmin, xmax = calculateMinMaxPoints(font, original_img_preview, 
                                            original_img_previewHeight, original_img_previewWidth, approx)
                
                textYMinMax = "green ymin: " + str(ymin) + " ymax: " + str(ymax)
                textXMinMax = "green xmin: " + str(xmin) + " xmax: " + str(xmax)
                cv2.putText(original_img_preview, textYMinMax, (0, 100), 
                            font, 0.5, (0, 0, 255)) 
                cv2.putText(original_img_preview, textXMinMax, (0, 120), 
                            font, 0.5, (0, 0, 255)) 
                #break

    for cnt in contoursInBlueMask:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(original_img_preview, [approx], 0, (100, 0, 200), 1)

        #print('len contours2 %d' %len(contours2)) 
        if len(approx) == 4:
            #print('len approx %d' %len(approx))
            if area > areaThreashold : 
                #print('area %d' %area)
                cv2.drawContours(original_img_preview, [approx], 0, (14, 59, 255), 2)

                ymin, ymax, xmin, xmax = calculateMinMaxPoints(font, original_img_preview, 
                                            original_img_previewHeight, original_img_previewWidth, approx)
                
                textYMinMax = "blue ymin: " + str(ymin) + " ymax: " + str(ymax)
                textXMinMax = "blue xmin: " + str(xmin) + " xmax: " + str(xmax)
                cv2.putText(original_img_preview, textYMinMax, (0, 140), 
                            font, 0.5, (0, 0, 255)) 
                cv2.putText(original_img_preview, textXMinMax, (0, 160), 
                            font, 0.5, (0, 0, 255))
                #break

    lowerThirdYUpper = 2 * int(original_img_previewHeight/3)
    line_thickness = 2
    cv2.line(original_img_preview, (0, lowerThirdYUpper), (original_img_previewWidth, lowerThirdYUpper), (0, 255, 0), thickness=line_thickness)
    
    lowerEightXLower = int(original_img_previewWidth/9.1)
    line_thickness = 2
    cv2.line(original_img_preview, (lowerEightXLower, lowerThirdYUpper), (lowerEightXLower, original_img_previewHeight), (0, 255, 0), thickness=line_thickness)

    lowerEightXUpper = int(8.1 * int(original_img_previewWidth/9.1))
    line_thickness = 2
    cv2.line(original_img_preview, (lowerEightXUpper, lowerThirdYUpper), (lowerEightXUpper, original_img_previewHeight), (0, 255, 0), thickness=line_thickness)

    cv2.imshow('green mask', green_mask_half)
    cv2.imshow('blue mask', blue_mask_half)
    cv2.imshow('Original preview', original_img_preview)

    #cv2.imshow('Original full + contours', image)
    #cv2.imshow("image", mask)

    key = cv2.waitKey(1)
    if key == 27: # ESC
        break

#cv2.imwrite("mask-test.jpg", mask)     # save frame as JPEG file 
#cv2.imwrite("mask-original-test.jpg", image)

print('Done.')