import cv2

stream = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

while(True):
  (grabbed, frame) = stream.read()
  
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.3, minNeighbors=5)
  
  for (x,y,w,h) in faces:
    color = (0,255,255)
    stroke = 5
    cv2.rectangle(frame, (x,y), (x+w, y+h), color, stroke)


  cv2.imshow("Image", frame)
  key = cv2.waitKey(1)&0xFF
  if key == ord("q"):
    break

stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)