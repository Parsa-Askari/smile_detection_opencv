import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

smile_cascade=cv.CascadeClassifier("./xml/haarcascade_smile.xml")
face_cascade=cv.CascadeClassifier("./xml/haarcascade_frontalface_default.xml")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    face_list=[]
    face_cords=[]
    smiles=[]
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for (x,y,w,h) in faces:
        face_list.append(gray[y:y+h,x:x+w])
        face_cords.append([x,y])
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=1)
    
    for i,face in enumerate(face_list):
        smiles=smile_cascade.detectMultiScale(face,scaleFactor=1.1,minNeighbors=200)
        x_base=face_cords[i][0]
        y_base=face_cords[i][1]
        for (x,y,w,h) in smiles:
            cv.rectangle(frame,(x_base+x,y_base+y),(x_base+x+w,y_base+y+h),(255,0,0),thickness=1)


    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()