import cv2
import numpy as np
from utils import (load_best_model)
from keras.utils import img_to_array

emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear','contempt', 'unknown']
# Load model and weights from saved directory
model = load_best_model()
print("Successfully loaded FER Model!")

# Use pretrained model for face detection
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vid = cv2.VideoCapture(0)
print("Successfully loaded Face Detection Model!")

while True:
    ret, img = vid.read()
    if not ret:
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_heads = haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(48, 48))

    for (x, y, w, h) in detected_heads:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        head = gray_img[y:y + w, x:x + h]
        head = cv2.resize(head, (48, 48))
        head_pixels = img_to_array(head)
        head_pixels = np.expand_dims(head_pixels, axis=0)
        head_pixels /= 255.0

        pred = model.predict(head_pixels)
        max_index = int(np.argmax(pred))

        predicted_emotion = emotions[max_index]

        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        resized_img = cv2.resize(img, (1000, 700))
        cv2.imshow('Facial Emotion Recognition', resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
