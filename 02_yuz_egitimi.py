

import cv2
import numpy as np
from PIL import Image
import os

# path of data
path = 'veriseti'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml");

# function for importing and tagging images
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    ornekler=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')    # grey
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[0])
        # print("id= ",id)
        yuzler = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in yuzler:
            ornekler.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return ornekler,ids
print ("\n [INFO] yuzler eğitiliyor. Birkaç saniye bekleyin ...")
yuzler,ids = getImagesAndLabels(path)
recognizer.train(yuzler, np.array(ids))
# Save the model to the file egitim/egitim.xml
recognizer.write('egitim/egitim.yml')
# Show the trained face count and finalize the code
print(f"\n [INFO] {len(np.unique(ids))} yüz eğitildi. Betik sonlandırılıyor.")

# print(yuzler)