
import cv2

kamera = cv2.VideoCapture(0)
kamera.set(3, 640) 
kamera.set(4, 420) 
face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
# Assign a different face integer for each different person
face_id = input('\n enter user id end press <return> ==>  ')
MAXFOTOSAY = 50 # Number of images to use for each face
#2face_id = 1
print("\n [INFO] Kayıtlar başlıyor. Kameraya bak ve bekle ...")

count = 0

while(True):
    ret, img = kamera.read()
    #img = cv2.flip(img, -1) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yuzler = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in yuzler:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        # Save the captured image to the dataset folder
        cv2.imwrite("veriseti/" + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('imaj', img)
        print("Kayıt no: ",count)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= MAXFOTOSAY:
         break
# Clear memory
print("\n [INFO] Program sonlanıyor ve bellek temizleniyor.")
kamera.release()
cv2.destroyAllWindows()