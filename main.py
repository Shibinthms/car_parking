import cv2
import pickle
import cvzone #computer vision package..
import numpy as np

video = cv2.VideoCapture("carPark.mp4")

with open("CarParkPos", 'rb') as f:
    posList = pickle.load(f)#: Opens the file "CarParkPos" in binary mode for reading. It loads the previously saved posList using pickle.

width, height = 107, 48#These values are used for defining the size of rectangles and other purposes.

def checkParkinSpace(imgPro):#: Defines a function checkParkingSpace that takes a processed image (imgPro) as input.
    spacecounter = 0# Initializes a counter variable for counting parking spaces.
    for pos in posList:
        x, y = pos## Retrieve the x and y coordinates from the current position
        imgcrop = imgPro[y:y+height, x:x+width]
        
        
        
# Counts the number of non-zero pixels in the ROI using cv2.countNonZero. This is used to determine whether a parking space is occupied or not.
# Updates the color and thickness of the rectangle based on whether the space is occupied or not.
# Draws rectangles on the original image (img) based on the positions and conditions.
# Displays the count inside each rectangle.
# Updates the space counter.
        count = cv2.countNonZero(imgcrop)
        if count > 4900:
            color = (0, 255, 0)
            thickness = 5
            spacecounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        font_scale = min(width, height) / 1000.0
        
        # Display the count inside each rectangle
        cv2.putText(img, str(count), (pos[0] + 10, pos[1] + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2)
        
    # Display the number of free spaces
    cvzone.putTextRect(img, f'Free: {spacecounter}/{len(posList)}', (100, 50),
                       scale=2, thickness=3, offset=20, colorB=(255, 0, 0), border=2)

while True:
    if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
        #Checks if the current frame index is equal to the total number of frames in the video.
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = video.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernal = np.ones((3, 3), np.int8)
    imgDilate = cv2.dilate(imgMedian, kernal, iterations=1)
    #Dilation is a morphological operation that enhances the bright regions in an image

    checkParkinSpace(imgDilate)

    cv2.imshow("VIDEO", img)
    # cv2.imshow("BLUR", imgBlur)
    # cv2.imshow("imgThresh", imgMedian)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

