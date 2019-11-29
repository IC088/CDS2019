import cv2
import math

videoFile = "scene1.mp4"

cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
print(frameRate)
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate*5) == 0):
        print(f'Saving {str(int(x))} now')
        filename = str(int(x)) + ".jpg"
        x+=1
        cv2.imwrite(filename, frame)
        # cv2.show(frame)

cap.release()
print ("Done!")