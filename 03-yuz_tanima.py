
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def print_utf8_text(image, xy, text, color): 
    fontName = 'arial.ttf'  
    font = ImageFont.truetype(fontName, 24)  
    img_pil = Image.fromarray(image)  
    draw = ImageDraw.Draw(img_pil)  
    draw.text((xy[0],xy[1]), text, font=font,
              fill=(color[0], color[1], color[2], 0))  # b,g,r,a
    image = np.array(img_pil)  
    return image

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('egitim/egitim.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['None', 'semih','ferad']

kamera = cv2.VideoCapture(0)
kamera.set(3, 640)  
kamera.set(4, 480)  

minW = 0.1*kamera.get(3)  
minH = 0.1*kamera.get(4)  
while True:
    ret, img = kamera.read()
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    yuzler = faceCascade.detectMultiScale(
        gri,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in yuzler:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, uyum = recognizer.predict(gri[y:y + h, x:x + w])

        if (uyum < 100):
            id = names[id]
            uyum = "  {0}%".format(round(100 - uyum))
        else:
            id = "bilinmiyor"
            uyum = "  {0}%".format(round(100 - uyum))

        color = (255,255,255)
        img=print_utf8_text(img,(x+5,y-25),str(id),color)
        # cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(uyum), (x + 5, y + h + 25), font, 1, (255, 255, 0), 1)

    cv2.imshow('kamera', img)
    k = cv2.waitKey(10) & 0xff  # For exit Esc or q 
    if k == 27 or k==ord('q'):
        break
# Belleği temizle
print("\n [INFO] Programdan çıkıyor ve ortalığı temizliyorum")
kamera.release()
cv2.destroyAllWindows()