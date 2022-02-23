import cv2
import numpy as np

def sketch(image) :
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
    edges = cv2.Canny(img_blur,30,50)
    ret, mask = cv2.threshold(edges,70,255,cv2.THRESH_BINARY)
    return mask

cap = cv2.VideoCapture(0)

while True :
    ret, frame = cap.read()
    cv2.imshow('Sketch', sketch(frame))
    if cv2.waitKey(1) == 0 :
        break

cap.release()
cv2.destroyAllWindows()