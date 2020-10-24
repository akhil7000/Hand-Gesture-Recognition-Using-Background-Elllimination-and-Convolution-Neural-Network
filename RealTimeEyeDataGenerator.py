import numpy as np
import cv2
from PIL import Image

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((89,100), Image.ANTIALIAS)
    print(img.size)
    img.save(imageName)


def main():


    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
    
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    cap = cv2.VideoCapture(0)
    
    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            counter=1;
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                roi_eyes_gray = roi_color[ey:ey+eh, ex:ex+ew]
                roi_eyes_color = roi_color[ey:ey+eh, ex:ex+ew]
                cv2.imwrite('Dataset/eye'+str(counter)+'.png',roi_eyes_color)
                resizeImage('Dataset/eye'+str(counter)+'.png')
                counter=counter+1
    
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
main()

    
    