# import cv2
# import pickle
# from imutils.video import WebcamVideoStream
# import face_recognition
# # img=WebcamVideoStream(src=0).start().read()
# # print(img)
#
# class VideoCamera(object):
#     def __init__(self):
#
#         self.stream = WebcamVideoStream(src=0).start()
#
#     def __del__(self):
#         self.stream.stop()
#
#     # def predict(self, frame, knn_clf, distance_threshold=0.4):
#     #     # Find face locations
#     #     X_face_locations = face_recognition.face_locations(frame)
#     #     # print("X_face_locations",X_face_locations[0])
#     #     # X_face_locations[0][0]: X_face_locations[0][1], X_face_locations[0][2]: X_face_locations[0][3]
#     #     # try:
#     #     #     print("here")
#     #     #     cv2.imshow("fdgd",frame[57:304,242:118])
#     #     #     cv2.waitKey(1)
#     #     # except:
#     #     #     pass
#     #     # If no faces are found in the image, return an empty result.
#     #     if len(X_face_locations) == 0:
#     #         return []
#     #
#     #     # Find encodings for faces in the test iamge
#     #     faces_encodings = face_recognition.face_encodings(frame, known_face_locations=X_face_locations)
#     #
#     #     # Use the KNN model to find the best matches for the test face
#     #     closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
#     #     are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
#     #     for i in range(len(X_face_locations)):
#     #         print("closest_distances")
#     #         print(closest_distances[0][i][0])
#     #
#     #     # Predict classes and remove classifications that aren't within the threshold
#     #     return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
#     #             zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
#
#     def get_frame(self):
#         image = self.stream.read()
#
#         detector=cv2.CascadeClassifier('/home/ashish/Python_Machine_Learning_software/data/haarcascades_GPU/haarcascade_frontalface_default.xml')
#         face=detector.detectMultiScale(image,1.1,7)
#         [cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2) for (x,y,h,w) in face]
#         ret, jpeg = cv2.imencode('.jpg', image)
#         data = []
#         data.append(jpeg.tobytes())
#         return data

import cv2
import pickle
from imutils.video import WebcamVideoStream
import pygame
import numpy as np
import glob
import datetime as dt

MIN_MATCH = 20
showText = ''

pygame.mixer.init()

pygame.mixer.set_num_channels(8)

player = pygame.mixer.Channel(2)

sound20 = pygame.mixer.Sound('sound/20.ogg')
sound50 = pygame.mixer.Sound('sound/50.ogg')
sound100 = pygame.mixer.Sound('sound/100.ogg')
sound200 = pygame.mixer.Sound('sound/200.ogg')
sound500 = pygame.mixer.Sound('sound/500.ogg')

def load_data_set():
            
            data_bill_100_1=[img[0],temp_kp[0], temp_desc[0],'100', sound100]
            data_bill.append(data_bill_100_1)
            data_bill_100_2=[img[1],temp_kp[1], temp_desc[1],'100', sound100]
            data_bill.append(data_bill_100_2)
            data_bill_100_3=[img[2],temp_kp[2], temp_desc[2],'100', sound100]
            data_bill.append(data_bill_100_3)
            data_bill_100_4=[img[3],temp_kp[0], temp_desc[0],'100', sound100]
            data_bill.append(data_bill_100_4)

            data_bill_100n_1=[img[4],temp_kp[4], temp_desc[4],'100n', sound100]
            data_bill.append(data_bill_100n_1)
            data_bill_100n_2=[img[5],temp_kp[5], temp_desc[5],'100n', sound100]
            data_bill.append(data_bill_100n_2)
            data_bill_100n_3=[img[6],temp_kp[6], temp_desc[6],'100n', sound100]
            data_bill.append(data_bill_100n_3)
            data_bill_100n_4=[img[7],temp_kp[7], temp_desc[7],'100n', sound100]
            data_bill.append(data_bill_100n_4)

            data_bill_200_1=[img[8],temp_kp[8], temp_desc[8], '200', sound200]
            data_bill.append(data_bill_200_1)
            data_bill_200_2=[img[9],temp_kp[9], temp_desc[9], '200', sound200]
            data_bill.append(data_bill_200_2)
            data_bill_200_3=[img[10],temp_kp[10], temp_desc[10], '200', sound200]
            data_bill.append(data_bill_200_3)
            data_bill_200_4=[img[11],temp_kp[11], temp_desc[11], '200', sound200]
            data_bill.append(data_bill_200_4)

            data_bill_20_1=[img[12],temp_kp[12], temp_desc[12], '20', sound20]
            data_bill.append(data_bill_20_1)
            data_bill_20_2=[img[13],temp_kp[13], temp_desc[14], '20', sound20]
            data_bill.append(data_bill_20_2)
            data_bill_20_3=[img[14],temp_kp[14], temp_desc[14], '20', sound20]
            data_bill.append(data_bill_20_3)
            data_bill_20_4=[img[15],temp_kp[15], temp_desc[15], '20', sound20]
            data_bill.append(data_bill_20_4)

            data_bill_500_1=[img[16],temp_kp[16], temp_desc[16], '500', sound500]
            data_bill.append(data_bill_500_1)
            data_bill_500_2=[img[17],temp_kp[17], temp_desc[17], '500', sound500]
            data_bill.append(data_bill_500_2)
            data_bill_500_3=[img[18],temp_kp[18], temp_desc[18], '500', sound500]
            data_bill.append(data_bill_500_3)
            data_bill_500_4=[img[19],temp_kp[19], temp_desc[19], '500', sound500]
            data_bill.append(data_bill_500_4)

            data_bill_500n_1=[img[20],temp_kp[20], temp_desc[20], '500n', sound500]
            data_bill.append(data_bill_500n_1)
            data_bill_500n_2=[img[21],temp_kp[21], temp_desc[21], '500n', sound500]
            data_bill.append(data_bill_500n_2)
            data_bill_500n_3=[img[22],temp_kp[22], temp_desc[22], '500n', sound500]
            data_bill.append(data_bill_500n_3)
            data_bill_500n_4=[img[23],temp_kp[23], temp_desc[23], '500n', sound500]
            data_bill.append(data_bill_500n_4)

            data_bill_50_1=[img[24],temp_kp[24], temp_desc[24], '50', sound50]
            data_bill.append(data_bill_50_1)
            data_bill_50_2=[img[25],temp_kp[25], temp_desc[25], '50', sound50]
            data_bill.append(data_bill_50_2)
            data_bill_50_3=[img[26],temp_kp[26], temp_desc[26], '50', sound50]
            data_bill.append(data_bill_50_3)
            data_bill_50_4=[img[27],temp_kp[27], temp_desc[27], '50', sound50]
            data_bill.append(data_bill_50_4)


def preprocess (frame):
    frame = cv2.bilateralFilter(frame, 6, 60, 60)
    #frame = cv2.blur(frame,(6,6))
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
    #pygame.time.wait(1000)
    #pygame.mixer.stop()

#Función para detección de la imagen/matcheo
def match_draw(img1,img2,found,cont,showText,sound):

    font = cv2.FONT_HERSHEY_SIMPLEX

    cont =+ 1

    kp1, desc1 = detector.detectAndCompute(img1, None)
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
        cont = cont + 1
        #print(cont)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        accuracy = float(mask.sum()) / mask.size
        #print("accuracy: %d/%d(%.2f%%)" % (mask.sum(), mask.size, accuracy))

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
        cont = 0

    
    #---------------------------------------
    #solo resultado de la detección
    
    #---------------------------------------
    
    #resultado de matches y comparación
    #res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
    #                      matchesMask=matchesMask,
    #                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    #res = img2
    #cv2.imshow("prueba", res)
    #res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                          #matchesMask=matchesMask,
                          #flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    #cv2.putText(res, showText, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
    #cv2.imshow("hello", res)
    return found, cont

#cargar imágenes y llenar arrays con feautures como referencias para la comparación de imágenes
#recibe los array de imagenes y features globales
def load(temp_kp, temp_desc,img):

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

#Inicializa el detector, matcher y los arrays de features y set de imágenes
detector, matcher = init_feature()
temp_kp = []
temp_desc = []
img = []


load(temp_kp, temp_desc,img)

class VideoCamera(object):
    

    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.

        self.stream = WebcamVideoStream(src=0).start()
        

    def __del__(self):
        self.stream.stop()

   
    @with_goto
    def get_frame(self,cont,found,searchIndex,sound):
        
        while(True):
            label .find
            #print(searchIndex)
            #comienza la comparación para detectar el billete
            if not found:
                #print("inicio")
                if searchIndex <= 28:
                    if searchIndex == 1:
                        img1 = img[0]
                        kp1 = temp_kp[0]
                        desc1 = temp_desc[0]
                        showText = '100'
                        print(showText)
                    elif searchIndex == 2:
                        img1 = img[1]
                        kp1 = temp_kp[1]
                        desc1 = temp_desc[1]
                        showText = '100'
                    elif searchIndex == 3:
                        img1 = img[2]
                        kp1 = temp_kp[2]
                        desc1 = temp_desc[2]
                        showText = '100'
                    elif searchIndex == 4:
                        img1 = img[3]
                        kp1 = temp_kp[3]
                        desc1 = temp_desc[3]
                        showText = '100'
                    elif searchIndex == 5:
                        img1 = img[4]
                        kp1 = temp_kp[4]
                        desc1 = temp_desc[4]
                        showText = '100'
                    elif searchIndex == 6:
                        img1 = img[5]
                        kp1 = temp_kp[5]
                        desc1 = temp_desc[5]
                        showText = '100'
                    elif searchIndex == 7:
                        img1 = img[6]
                        kp1 = temp_kp[6]
                        desc1 = temp_desc[6]
                        showText = '100'
                    elif searchIndex == 8:
                        img1 = img[7]
                        kp1 = temp_kp[7]
                        desc1 = temp_desc[7]
                        showText = '100'

                    #200
                    elif searchIndex == 9:
                        img1 = img[8]
                        kp1 = temp_kp[8]
                        desc1 = temp_desc[8]
                        showText = '200'
                        print(showText)
                    elif searchIndex == 10:
                        img1 = img[9]
                        kp1 = temp_kp[9]
                        desc1 = temp_desc[9]
                        showText = '200'
                    elif searchIndex == 11:
                        img1 = img[10]
                        kp1 = temp_kp[10]
                        desc1 = temp_desc[10]
                        showText = '200'
                    elif searchIndex == 12:
                        img1 = img[11]
                        kp1 = temp_kp[11]
                        desc1 = temp_desc[11]
                        showText = '200'

                    #20
                    elif searchIndex == 13:
                        img1 = img[12]
                        kp1 = temp_kp[12]
                        desc1 = temp_desc[12]
                        showText = '20'
                        print(showText)
                    elif searchIndex == 14:
                        img1 = img[13]
                        kp1 = temp_kp[13]
                        desc1 = temp_desc[13]
                        showText = '20'
                    elif searchIndex == 15:
                        img1 = img[14]
                        kp1 = temp_kp[14]
                        desc1 = temp_desc[14]
                        showText = '20'
                    elif searchIndex == 16:
                        img1 = img[15]
                        kp1 = temp_kp[15]
                        desc1 = temp_desc[15]
                        showText = '20'

                    #500
                    elif searchIndex == 17:
                        img1 = img[16]
                        kp1 = temp_kp[16]
                        desc1 = temp_desc[16]
                        showText = '500'
                        print(showText)
                    elif searchIndex == 18:
                        img1 = img[17]
                        kp1 = temp_kp[17]
                        desc1 = temp_desc[17]
                        showText = '500'
                    elif searchIndex == 19:
                        img1 = img[18]
                        kp1 = temp_kp[18]
                        desc1 = temp_desc[18]
                        showText = '500'
                    elif searchIndex == 20:
                        img1 = img[19]
                        kp1 = temp_kp[19]
                        desc1 = temp_desc[19]
                        showText = '500'
                    elif searchIndex == 21:
                        img1 = img[20]
                        kp1 = temp_kp[20]
                        desc1 = temp_desc[20]
                        showText = '500'
                    elif searchIndex == 22:
                        img1 = img[21]
                        kp1 = temp_kp[21]
                        desc1 = temp_desc[21]
                        showText = '500'
                    elif searchIndex == 23:
                        img1 = img[22]
                        kp1 = temp_kp[22]
                        desc1 = temp_desc[22]
                        showText = '500'
                    elif searchIndex == 24:
                        img1 = img[23]
                        kp1 = temp_kp[23]
                        desc1 = temp_desc[23]
                        showText = '500'

                    #50                
                    elif searchIndex == 25:
                        img1 = img[24]
                        kp1 = temp_kp[24]
                        desc1 = temp_desc[24]
                        showText = '50'
                        print(showText)
                    elif searchIndex == 26:
                        img1 = img[25]
                        kp1 = temp_kp[25]
                        desc1 = temp_desc[25]
                        showText = '50'
                    elif searchIndex == 27:
                        img1 = img[26]
                        kp1 = temp_kp[26]
                        desc1 = temp_desc[26]
                        showText = '50'
                    elif searchIndex == 28:
                        img1 = img[27]
                        kp1 = temp_kp[27]
                        desc1 = temp_desc[27]
                        showText = '50'
                    
                    #print("llegue aquí searchindex antes: ",searchIndex)
                    searchIndex = searchIndex+1
                    #print("llegue aquí searchindex despues: ",searchIndex)
                else:
                    showText = '100'
                    searchIndex = 1
                    img1 = img[0]
            
            #print("llegue a escoger sonido")
            #escoger el sonido a reproducir 
            if showText == '20':
                sound = sound20
            elif showText == '50':
                sound = sound50
            elif showText == '100':
                sound = sound100
            elif showText == '200':
                sound = sound200
            elif showText == '500':
                sound = sound500
            
            img2 = self.stream.read()
            img2 = preprocess(img2)

            found, cont = match_draw(img1, img2, found, cont, showText,sound)
            
            
            if found:
                ret, jpeg = cv2.imencode('.jpg', img2)
                data = []
                data.append(jpeg.tobytes())
                return data
            else:
                goto .find


        

        
    

        #cv2.putText(img2, showText, (100 - 70, 200 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        
        
