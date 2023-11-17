import cv2
import mediapipe as mp
from numpy import arange


mp_facemash = mp.solutions.face_mesh
face_mesh = mp_facemash.FaceMesh()

points_vector = [243,130,359,463,10,331,102,1,291,61,6,168,164] #,0
#
cap = cv2.VideoCapture(0)

while True:
    res,image = cap.read()
    if res:
        #image = cv2.imread('prova1.png')
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        h,w,_ = image.shape
        
        results = face_mesh.process(image)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks is not None:
            for i in results.multi_face_landmarks:
                for j in points_vector:
                    pt = i.landmark[j]
                    x = int(pt.x * w)
                    y = int(pt.y * h)
                    cv2.circle(image,(x,y),3,(255,0,0),-1)
            
        cv2.imshow('faccia',image)
        cv2.waitKey(1)
        