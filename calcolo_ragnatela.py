import os
import time
import threading
import multiprocessing
import math
from pylab import *
import PIL.Image as im
import csv
import sys

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from string import Template
import string

def distanza(x1, y1, x2, y2):
    x12 = (x2 - x1) * (x2 - x1)
    y12 = (y2 - y1) * (y2 - y1)
    xy = x12 + y12
    dist = math.sqrt(xy)
    return dist



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

intero = 0

voltostr = ''


cerchi = 4
fetteQ = 4  # fette per quadrante

fette = fetteQ * 4
s1 = cerchi * fette

#dizionario = np.zeros((2223, s1))

dizionario = [ [ 0 for y in range(s1)] for x in range(2230)]
    #x = width
    #y = height

dizionario_str = ['' for xx in range(2230)]

volto = np.zeros(s1)



def aggiungi(xcentro, ycentro, rax, xpunto, ypunto, distNaso, coeff, immm):

    indice = 0

    settore = np.zeros(3) #cerchio, quadrante, fetta

    # distNaso =  distanza dal naso

    a = 0  # a = raggioStart
    b8 = 4 * rax / 10  # b = raggioStop
    i = 1  # in quale cerchio cade il punto. i = [1, cerchi]

    b4 = 7 * rax / 10
    b2 = 9 * rax / 10

    #cerchi
    if( distNaso > a and distNaso <= b8):
        settore[0] = 1
    elif(distNaso > b8 and distNaso <= b4):
        settore[0] = 2
    elif(distNaso > b4 and distNaso <= b2):
        settore[0] = 3
    else :
        settore[0] = 4


    #quadrante
    if (xpunto <= xcentro and y <= ycentro):
        # il punto appartiene al quadrante in alto a sinistra
        settore[1] = 2
    elif (x <= xnose and y >= ynose):
        # il punto appartiene al quadrante in basso a sinistra
        settore[1] = 3
    elif (x >= xnose and y <= ynose):
        # il punto appartiene al quadrante in alto a destra
        settore[1] = 1
    else:
        # il punto appartiene al quadrante in basso a destra
        settore[1] = 4

    a = 0                 #grado Start
    b = 90  / fetteQ      #grado Stop
    i = 1                 #in quale fetta cade il punto. i = [1, fette]

    radang_a = 0                    # radiante Start
    radang_b = math.radians(b)      # radiante Stop
    tng_a = math.tan(radang_a)
    tng_b = math.tan(radang_b)


    #fetta
    while(settore[2] == 0 and b < 90):
        if(coeff > tng_a and coeff <= tng_b):
            settore[2] = i
        b = b + (90  / fetteQ)
        radang_b = math.radians(b)  # radiante Stop
        tng_a = tng_b
        tng_b = math.tan(radang_b)
        i = i+1

    if(xpunto == xnose):
        settore[2] = 1


    if(settore[2] == 0):
        settore[2] = fetteQ

        # settore[0] = cerchio
        # settore[1] = quadrante
        # settore[2] = fetta

    if(settore[1] == 1 or settore[1] == 3):
        indice = int(fette * (settore[0]-1) + fetteQ * (settore[1] -1) + abs(settore[2] - 4 ) -1)
    else:
        indice = int(fette * (settore[0] - 1) + fetteQ * (settore[1] - 1) + settore[2] - 1)


    try:
        if (xnose != xpunto or ynose != ypunto):   #il naso non ha settore
            volto[indice] = int(volto[indice] + 1)
    except:
        #else:
        print("ERROOOOOOOREEEEEE------")
        print("indice ", indice)

        #print("xnose ", xnose, " xpunto ", xpunto, " ynose ", ynose , " ypunto " , ypunto)

    return indice


immagini = os.listdir('DatasetCelebA')

num_volto = 0

for img in immagini:
    if img.find(".jpg") > 0:

        tick_detector = time.time()

        im2 = "DatasetCelebA/"+str(img)

        foto = cv2.imread(im2)

        volto = np.zeros(s1)

        xnose = 0
        ynose = 0
        raggio = 0

        xlont = 0
        ylont = 0

        foto = imutils.resize(foto, width=512)
        gray = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)

        tick_detector = time.time()

        rects = detector(foto, 1)

        #print("tick detector " , time.time() - tick_detector)

        dista = 0
        raggio = 0

        m = 0
        d = 0
        n = 1
        imga = zeros([512, 512, 3])
        for (i, rect) in enumerate(rects):

            tick_predictor = time.time()

            shape = predictor(gray, rect)

            #print("tick predictor ", time.time() - tick_predictor)

            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            xnose = shape[33][0]
            ynose = shape[33][1]

            for (x, y) in shape:

                tick_volto = time.time()

                dista = distanza(xnose, ynose, x, y)
                if(dista > raggio) :
                    raggio = dista
                    xlont = x   #coordinata x del punto più lontano dal naso
                    ylont = y   #coordinata y del punto più lontano dal naso
            for(x,y) in shape:
                settore = [0,0,0]
                if(y == ynose):
                    m = 0
                else:
                    m = (x - xnose)/(y-ynose)
                m = abs(m)

                d = distanza(xnose, ynose, x,y)

                tick_punto = time.time()

                nnn = aggiungi(xnose, ynose, raggio, x, y, d, m, imga )

                #print("tick punto " , time.time() - tick_punto)

                ## E SE è LA PUNTA DEL NASO ?
                #try:
                #except:
                #print(num_volto)
                #dizionario stringa = nome immagine

                #nstr = str(nnn)

                #if (x == xnose and y == ynose):
                    #nstr = "0"

                #cv2.putText(imga, nstr, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        dizionario[num_volto] = volto

        #print("-----------------------------------------")

        #print("tick_volto = ",  time.time() - tick_volto)

        nomeimagine = str(img)
        nomeimagine = nomeimagine[:15]
        nomeimagine = nomeimagine[6:]

        dizionario_str[num_volto] = nomeimagine

        num_volto = num_volto +1
        if((num_volto % 200) == 0):
            print(num_volto)



print("dizionario = ", dizionario)
print("dizionario_str = ", dizionario_str)



#cv2.imwrite("gen_settore.jpg", imga)
#cv2.imshow('image',imga)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

#f = open('test.txt', 'w')
#f.close()




