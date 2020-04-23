#Importo la libreria
import cv2
#Carico il classificatore
face_cascade= cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
#Carico l'immagine
img = cv2.imread("test.jpg")
#Il classificare che stiamo utilizzando lavora esclusivamente su immagini in scala di grigi. Per ovviare a
#questo problema, ci basterà convertire la nostra immagine in questa rappresentazione.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Sfruttiamo il classificatore salvato in face_cascade, applicandolo all’immagine test, stampandone il contenuto
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
print (faces.shape)
print (faces)
#Disegno un rettangolo attorno al volto con gli elementi presenti in faces
#for (x, y, w, h) in faces:
    #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#cv2.imshow('img', img)
#cv2.waitKey()
#cv2.destroyAllWindows()
#cv2.imwrite("test_rectangle.png",img)

#Crop dei volti
cont = 0
for (x, y, w, h) in faces:
    img2 = img[ y:y+h , x:x+w, : ]
    #print (img.shape)
    #print (img2.shape)
    #print ("-------------")
    nome_img2 = 'volto'+str(cont)+".png"
    cv2.imwrite(nome_img2, img2)
    cont = cont + 1 