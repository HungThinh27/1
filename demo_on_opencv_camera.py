import cv2
from time import sleep
from imutils.video import VideoStream
import imutils
import time
import cv2

#set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0,255,0)
fontcolor1 = (0,0,255)

# load face detection weights
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load gender detection model
from tensorflow import keras
model = keras.models.load_model('./best_weight_model_vgg16.h5')

# Predict 
def predict(img):
    r_size = cv2.resize(img,(178,218))
    roi = cv2.cvtColor(r_size, cv2.COLOR_BGR2RGB)
    finish = roi.reshape(1,218,178,3)
    y_pre = model.predict(finish)
    result = str('Male' if y_pre > 0.5 else 'Female')
    print_result = f"Probability: {int(y_pre)} | Gender: {result}"
    print(print_result)
    return result
 
# Doc tu camera
# camera = cv2.VideoCapture(0)
camera = VideoStream(src=0).start()
time.sleep(1.0)

while (True):
    # Doc tu camera
    img = camera.read()
    # Resize de tang toc do xu ly
    img = imutils.resize(img, width=600)
    # Lat anh kh cho bi nguoc
    img = cv2.flip(img, 1)
    faces = face_cascade.detectMultiScale(img, 1.2, 10,minSize=(100,100))

    for (x, y, w, h) in faces:
        try:
            cut_img = img[y:y + h, x:x + w]
            result = predict(cut_img)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Gender: " + result, (x,y+h+30), fontface, fontscale, fontcolor ,2)
        except:
            print("Can't recognize face!")
            
    cv2.imshow("Picture", img)

    # Quit
    key = cv2.waitKey(1)
    if key==ord('q'):
        break

camera.release()
cv2.destroyAllWindows()