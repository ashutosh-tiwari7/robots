import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import load_model

def make_square(img) :
    if img.shape[1] == img.shape[0] :
        return img
    doublesize = cv2.resize(img,(2*img.shape[1], 2*img.shape[1]), interpolation = cv2.INTER_CUBIC)
    height = img.shape[0] *2
    width = img.shape[1]*2
    if (height > width):
        pad = int((height - width)/2)
        doublesize_square = cv2.copyMakeBorder(doublesize,0,0,pad,pad,cv2.BORDER_CONSTANT,value=[0,0,0])
    else:
        pad = (width - height)/2
        doublesize_square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
    return doublesize_square

def resize_img(dimensions, image) :
    dim = (24, 24)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=[0,0,0])
    return ReSizedImg

classifier = load_model('/home/diablo/fr/robots/py/models/mnist.h5')

print('Classify custom image (y/n):')
x = input()

if x == 'y' :
    print('Enter absolute path of the image:')
    x = input()
    image = cv2.imread(x)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    ker = np.ones((5,5),np.uint8)
    edged = cv2.dilate(edged,ker,iterations=1)
    contours, heir = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    full_number = []
    for c in contours :
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= 5 and h >= 25:
            roi = blurred[y:y + h, x:x + w]
            ret, roi = cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)
            roi = make_square(roi)
            roi = resize_img(28,roi)
            roi = roi.reshape(1,28,28,1)
            res = classifier.predict(roi, 1, verbose = 0)[0]
            classes = [0,1,2,3,4,5,6,7,8,9]
            class_labels=[classes[i] for i,prob in enumerate(res) if prob == 1.]
            full_number.append(class_labels[0])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, str(class_labels[0]), (x , y + 155), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("image", image)
            cv2.waitKey(0) 
    print(full_number)
    cv2.destroyAllWindows()
else :
    (x_train, y_train), (x_test, y_test)  = mnist.load_data()

    def draw_test(name, pred, input_im):
        BLACK = [0,0,0]
        expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, imageL.shape[0] ,cv2.BORDER_CONSTANT,value=BLACK)
        expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
        cv2.putText(expanded_image, str(pred), (152, 70) , cv2.FONT_HERSHEY_COMPLEX_SMALL,4, (0,255,0), 2)
        cv2.imshow(name, expanded_image)

    for i in range(0,10):
        rand = np.random.randint(0,len(x_test))
        input_im = x_test[rand]

        imageL = cv2.resize(input_im, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC) 
        input_im = input_im.reshape(1,28,28,1)
        res = classifier.predict(input_im, 1, verbose = 0)[0]
        classes = [0,1,2,3,4,5,6,7,8,9]
        class_labels=[classes[i] for i,prob in enumerate(res) if prob == 1.]
        
        draw_test("Prediction", class_labels[0], imageL) 
        cv2.waitKey(0)

    cv2.destroyAllWindows()