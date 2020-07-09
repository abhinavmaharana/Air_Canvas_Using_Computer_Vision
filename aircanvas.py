import numpy as np
import cv2
from collections import deque

# defining the upper and lower boundaries for a color to be considered "blue"
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# defining a 5x5 kernel for erosion and dilation
kernel = np.ones((5,5), np.uint8)

#seting up deques to store seperate colors in seperate arrays
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

bindex = 0
gindex = 0
rindex = 0
yindex = 0

colors = [(255, 0, 0), (0, 255, 0) , (0, 0, 255), (0,255,255)]
colorIndex = 0

#seting up the paint interface
paintWindow = np.zeros((471,636,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1),(140,65),(0,0,0),2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), colors[3], -1)
cv2.putText(paintWindow, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)

cv2.namedWindow('Paint' , cv2.WINDOW_AUTOSIZE)

# loading the video
camera = cv2.VideoCapture(0)

# keep looping 
while True:
    #grabbing the current paintwindow
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # adding the coloring options to the frame
    frame = cv2.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
    frame = cv2.rectangle(frame, (160,1),(255,65), colors[0], -1)
    frame = cv2.rectangle(frame, (275,1),(370,65), colors[1], -1)
    frame = cv2.rectangle(frame, (390,1),(485,65), colors[2], -1)
    frame = cv2.rectangle(frame, (505,1),(600,65), colors[3], -1)

    cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)


    #checking to see if we have reached the end of the video
    if not grabbed:
        break

    #determining which pixels fall within the blue boundaries and then blur the binary image
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel,iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)


    # Find contours in the image
    ( cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center = None


    # checking to see if any contours were found
    if len(cnts) > 0:
        #sorting the contours and finding the largest one
        cnt = sorted(cnts ,key = cv2.contourArea, reverse=True)[0]
        # geting the radius of the enclosing circle around the found contour
        ((x,y), radius) = cv2.minEnclosingCircle(cnt)
        #Draw the circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255),2)
        #geting the moments to calculate the center of the contour
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if center[1] <= 65:
            if 40 <= center[0] <=140:
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                bindex = 0
                gindex = 0
                rindex = 0
                yindex = 0

                paintWindow[67:,:,:] = 255
            
            elif 160 <= center[0] <= 255:
                colorIndex = 0 #blue color 
            elif 275 <= center[0] <= 370:
                colorIndex = 1  #green color
            elif 390 <= center[0] <= 485:
                colorIndex = 2 #Red color
            elif 505 <= center[0] <= 600:
                colorIndex = 3 #yellow color

        else:
            
            if colorIndex == 0:
                bpoints[bindex].appendleft(center)
            elif colorIndex == 1:
                gpoints[gindex].appendleft(center)
            elif colorIndex == 2:
                rpoints[rindex].appendleft(center)
            elif colorIndex == 3:
                ypoints[yindex].appendleft(center)       

        #Draw lines of all the colors(Blue, Green, Red and Yellow)
        points = [bpoints,gpoints,rpoints,ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i],2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i],2)

        
        #showing the frame and the paintWindow image
        cv2.imshow("VideoLive", frame)
        cv2.imshow("AirCanvas", paintWindow)

        #if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

#cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()    

