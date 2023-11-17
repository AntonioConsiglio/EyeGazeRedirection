import cv2
import mediapipe as mp
import numpy as np

mp_facemash = mp.solutions.face_mesh
face_mesh = mp_facemash.FaceMesh()

# points_vector = [243,130,359,463,10,331,102,1,291,61,6,168,164] #,0
points_vector = [[225,221,128,31],[441,445,261,453]] #,0

def get_max_min(list):
    return max(list),min(list)
#
cap = cv2.VideoCapture(1)

EYE_SHAPE = 150

while True:
    res,image = cap.read()
    if res:
        #image = cv2.imread('prova1.png')
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        h,w,_ = image.shape
        righteye = lefteye = np.zeros((EYE_SHAPE,EYE_SHAPE,3))
        
        results = face_mesh.process(image)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks is not None:
            for i in results.multi_face_landmarks:
                for n,landmark in enumerate(points_vector):

                    landmarp = [i.landmark[k] for k in landmark]

                    xmax,xmin = get_max_min([xp.x for xp in landmarp])
                    ymax,ymin = get_max_min([yp.y for yp in landmarp])

                    xmax,xmin = int(xmax * w),int(xmin * w)
                    ymax,ymin = int(ymax * h),int(ymin * h)

                    imw,imh = xmax-xmin,ymax-ymin
                    left = (EYE_SHAPE - imw )// 2
                    right = EYE_SHAPE - imw -left

                    top = (EYE_SHAPE - imh )// 2
                    bottom = EYE_SHAPE - imh - top

                    if n == 1:
                        righteye = image[ymin:ymax,xmin:xmax,:]
                        righteye = cv2.resize(cv2.copyMakeBorder(righteye, top, bottom, left, right, cv2.BORDER_CONSTANT),(0,0),fx=2,fy=2)
                    else:
                        lefteye = image[ymin:ymax,xmin:xmax,:]
                        lefteye = cv2.resize(cv2.copyMakeBorder(lefteye, top, bottom, left, right, cv2.BORDER_CONSTANT),(0,0),fx=2,fy=2)


                    cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color=(255,0,0),thickness=2)
            
        cv2.imshow('faccia',image)
        cv2.imshow("right",righteye)
        cv2.imshow("left",lefteye)
        cv2.waitKey(1)
        