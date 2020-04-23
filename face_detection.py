import cv2
face_cascade= cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
img = cv2.imread("test.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("volti_grigi.jpg",gray)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
print (faces.shape)
print (faces)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#cv2.imshow('img', img)
#cv2.waitKey()
#cv2.destroyAllWindows()
cv2.imwrite("test_rectangle.png",img)
cont = 0
for (x, y, w, h) in faces:
    img2 = img[ y:y+h , x:x+w, : ]
    #print (img.shape)
    #print (img2.shape)
    #print ("-------------")
    nome_img2 = 'volto'+str(cont)+".png"
    cv2.imwrite(nome_img2, img2)
    cont = cont + 1 