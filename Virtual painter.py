import cv2
import time
import numpy as np
import os
import HandTrackingModule as htm

#### Parameters ####
cam_width, cam_height = 640, 480

#### Video Capture ####
cap = cv2.VideoCapture(0) #Camera
cap.set(3, cam_width)
cap.set(4, cam_height)

#### Overlay ####
file_path = "proj_imgs/top_menu.png"
overlay_img = cv2.imread(file_path)

#### Drawing ####
draw_color = (255,0,0) #default blue
x_prev, y_prev = 0, 0
imgCanvas = np.zeros((480,640,3), np.uint8)


detector = htm.handDetector(detectionCon=0.80)

while True:
    success, img = cap.read()     #Hand image
    img = cv2.flip(img, 1)        #Flip horizontally
    img = detector.findHands(img)
    lm_list = detector.findPosition(img)

    if len(lm_list) != 0:

        #Get tips of index and middle fingers
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]

        fingers = detector.fingersUp() #Get a list with the value of each finger (0 or 1)

        #Selection mode - if 2 fingers up
        if fingers[1] and fingers[2]:
            x_prev, y_prev = 0, 0

            if y1 < 60:
                if 0 < x1 < 160:
                    draw_color = (0,0,255) #Red
                elif 161 < x1 < 320:
                    draw_color = (255,0,0) #Blue
                elif 321 < x1 < 480:
                    draw_color = (0,255,0)   #Green
                elif 481 < x1 < 640:
                    draw_color = (0,0,0)  #Black (erases)

            cv2.rectangle(img, (x1,y1-25), (x2, y2+25), (255,0,255), cv2.FILLED)

        #Drawing mode - if only index up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1), 15, draw_color, cv2.FILLED)

            if x_prev == 0 and y_prev == 0: #Avoid first frame draw
                x_prev, y_prev = x1, y1

            if draw_color == (255,255,255): #Eraser condition
                cv2.line(img, (x_prev, y_prev), (x1, y1), draw_color, 22)
                cv2.line(imgCanvas, (x_prev, y_prev), (x1, y1), draw_color, 22)
            else:
                cv2.line(img, (x_prev, y_prev), (x1, y1), draw_color, 12)
                cv2.line(imgCanvas, (x_prev, y_prev), (x1, y1), draw_color, 12)

            x_prev, y_prev = x1, y1

    #Overlay img
    overlay_img = cv2.resize(overlay_img, (640,60))
    img[0:60, 0:640] = overlay_img

    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Virtual Painter (Press Q to exit)", img)
    #cv2.imshow("Virtual Painter (Press Q to exit)", imgCanvas)
    if cv2.waitKey(1) == ord('q'): #shut down with q
        break