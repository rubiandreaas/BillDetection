import cv2
import pickle
from imutils.video import WebcamVideoStream
import pygame
import numpy as np
import glob
import datetime as dt


MIN_MATCH = 32
showText = ''
temp_kp = []
temp_desc = []
img = []
data_bill = []

pygame.mixer.init()

pygame.mixer.set_num_channels(8)

player = pygame.mixer.Channel(2)

sound20 = pygame.mixer.Sound('sound/20.ogg')
sound50 = pygame.mixer.Sound('sound/50.ogg')
sound100 = pygame.mixer.Sound('sound/100.ogg')
sound200 = pygame.mixer.Sound('sound/200.ogg')
sound500 = pygame.mixer.Sound('sound/500.ogg')


def preprocess (frame):
    frame = cv2.bilateralFilter(frame, 6, 60, 60)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(7, 7))
    frame = clahe.apply(frame)
    return frame

#Función para inicializar detector y matcher algoritmo ORB
def init_feature():
    detector = cv2.ORB_create(3500, 1.2, nlevels=9, edgeThreshold=31, firstLevel=0, WTA_K=2,
                              scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    return detector, matcher

#Función que reproduce el sonido que recibe como argumento
def play_sound(sound):
    pygame.init()
    sound.play()
    pygame.time.wait(1000)
    pygame.mixer.stop()

#Función para detección de la imagen/matcheo
def match_draw(img1,img2,found,showText,sound, kp, desc):

    font = cv2.FONT_HERSHEY_SIMPLEX

    kp1 = kp
    desc1 = desc 
    kp2, desc2 = detector.detectAndCompute(img2, None)
   

    matches = matcher.knnMatch(desc1, desc2, k=2)

    # Ordenar matches por distancia menor
    matches = sorted(matches, key=lambda x: x[0].distance)

    # Filtrar los mejores matches con umbral de distancia aceptable RADIO 0.8
    ratio = 0.8
    good_matches = [m[0] for m in matches
                    if len(m) == 2 and m[0].distance < m[1].distance * ratio]

    #print('good matches:%d/%d' % (len(good_matches), len(matches)))

    matchesMask = np.zeros(len(good_matches)).tolist()

    #Si al menos se encontraron 30 matches con buena coincidencia
    if len(good_matches) > MIN_MATCH:
        found = True

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        accuracy = float(mask.sum()) / mask.size
        print("accuracy %a: %d/%d(%.2f%%)  --- Good matches: %d/%d" % (showText, mask.sum(), mask.size, accuracy, len(good_matches), len(matches) ))

        #Si es posible que el conjuntos de puntos coincidentes
        #sean suficientes para dibujar el contorno del billete encontrado
        if mask.sum() > MIN_MATCH:

            matchesMask = mask.ravel().tolist()

            h, w, = img1.shape[:2]
            pts = np.float32([[[0, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w - 1, 0]]])
            dst = cv2.perspectiveTransform(pts, mtrx)
            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            #Escribe en pantalla la cantidad del billete encontrado
            cv2.putText(img2, showText, (20, 300), font, 10, (0, 255, 255), 10, cv2.LINE_4)

            #reproduce el sonido del billete encontrado si no hay otro sonido en cola de reproducción
            if pygame.mixer.get_busy() == False:
                play_sound(sound)
    else:
        found = False

   
    return found,


#Transfiere toda la info de las imágenes a un set de datos
def load_data_set():
    
            data_bill_500_1=[img[12],temp_kp[12], temp_desc[12], '500', sound500]
            data_bill.append(data_bill_500_1)
            data_bill_50_1=[img[16],temp_kp[16], temp_desc[16], '50', sound50]
            data_bill.append(data_bill_50_1)
            data_bill_100_1=[img[0],temp_kp[0], temp_desc[0],'100', sound100]
            data_bill.append(data_bill_100_1)
            data_bill_200_1=[img[4],temp_kp[4], temp_desc[4], '200', sound200]
            data_bill.append(data_bill_200_1)
            data_bill_20_1=[img[8],temp_kp[8], temp_desc[8], '20', sound20]
            data_bill.append(data_bill_20_1)
            data_bill_50_2=[img[17],temp_kp[17], temp_desc[17], '50', sound50]
            data_bill.append(data_bill_50_2)
            data_bill_100_2=[img[1],temp_kp[1], temp_desc[1],'100', sound100]
            data_bill.append(data_bill_100_2)
            data_bill_200_2=[img[5],temp_kp[5], temp_desc[5], '200', sound200]
            data_bill.append(data_bill_200_2)
            data_bill_20_2=[img[9],temp_kp[9], temp_desc[9], '20', sound20]
            data_bill.append(data_bill_20_2)
            data_bill_500_2=[img[13],temp_kp[13], temp_desc[13], '500', sound500]
            data_bill.append(data_bill_500_2)
            data_bill_50_3=[img[18],temp_kp[18], temp_desc[18], '50', sound50]
            data_bill.append(data_bill_50_3)
            data_bill_100_3=[img[2],temp_kp[2], temp_desc[2],'100', sound100]
            data_bill.append(data_bill_100_3)
            data_bill_200_3=[img[6],temp_kp[6], temp_desc[6], '200', sound200]
            data_bill.append(data_bill_200_3)
            data_bill_20_3=[img[10],temp_kp[10], temp_desc[10], '20', sound20]
            data_bill.append(data_bill_20_3)
            data_bill_500_3=[img[14],temp_kp[14], temp_desc[14], '500', sound500]
            data_bill.append(data_bill_500_3)
            data_bill_50_4=[img[19],temp_kp[19], temp_desc[19], '50', sound50]
            data_bill.append(data_bill_50_4)
            data_bill_100_4=[img[3],temp_kp[3], temp_desc[3],'100', sound100]
            data_bill.append(data_bill_100_4)
            data_bill_200_4=[img[7],temp_kp[7], temp_desc[7], '200', sound200]
            data_bill.append(data_bill_200_4)
            data_bill_20_4=[img[11],temp_kp[11], temp_desc[11], '20', sound20]
            data_bill.append(data_bill_20_4)
            data_bill_500_4=[img[15],temp_kp[15], temp_desc[15], '500', sound500]
            data_bill.append(data_bill_500_4)

#Inicializa el detector, matcher y los arrays de features y set de imágenes
detector, matcher = init_feature()

def load():

    # cargar la ruta de las imágenes
    filesname = [img for img in glob.glob("bills/*")]
    filesname.sort()

    # ordenar las rutas por orden alfabético
    filesname.sort()

    # recorrer todas las imágenes, calcular y guardar sus features en array
    for i in filesname:
        image = cv2.imread(i)
        img.append(image)
        t_temp_kp, t_temp_desc = detector.detectAndCompute(image, None)
        temp_kp.append(t_temp_kp)
        temp_desc.append(t_temp_desc)

load()
load_data_set()
img1 = img[0]

class VideoCamera(object):
    

    def __init__(self):
        self.stream = WebcamVideoStream(src=0).start()
        

    def __del__(self):
        self.stream.stop()

   
    def get_frame(self,found,sound):
    
        img2 = self.stream.read()
        img2 = preprocess(img2)

        data = []

        while(True):
            #comienza la comparación para detectar el billete
            for bill in data_bill:
                  
                img1 = bill[0]
                kp = bill[1]
                desc = bill[2]
                showText = bill[3]
                sound = bill[4]
                         
                found = match_draw(img1, img2, found, showText, sound, kp, desc)
                
            if found:
                ret, jpeg = cv2.imencode('.jpg', img2)
                data.append(jpeg.tobytes())
                return data