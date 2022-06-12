import cv2
from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import winsound
import os

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



lbl=['Close','Open']

model = load_model('models/cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL



class VideoCamera(object):
    
    def __init__(self):
        
        
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        count=0
        score=0
        thicc=2
        rpred=[99]
        lpred=[99]



#  success, image = self.video.read(0)
#         height,width = image.shape[0:2]
#         image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
#         gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#         faces=face_cascade.detectMultiScale(gray,1.3,5)
#         eyes= eye_cascade.detectMultiScale(gray, 1.3,5)

       
        ret, frame = self.video.read(0)
        height,width = frame.shape[:2] 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            # cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 3 )
             cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,0,0) , 5 )

        for (x,y,w,h) in right_eye:
           cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,0,0) , 2 )
           r_eye=frame[y:y+h,x:x+w]
           count=count+1
           r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
           r_eye = cv2.resize(r_eye,(24,24))
           r_eye= r_eye/255
           r_eye=  r_eye.reshape(24,24,-1)
           r_eye = np.expand_dims(r_eye,axis=0)
           rpred = model.predict_classes(r_eye)
           if(rpred[0]==1):
               
               lbl='Open' 
           if(rpred[0]==0):
            
            lbl='Closed'
           break

        for (x,y,w,h) in left_eye:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,0,0) , 2 )
            l_eye=frame[y:y+h,x:x+w]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = model.predict_classes(l_eye)
            if(lpred[0]==1):
               
               lbl='Open'   
            if(lpred[0]==0):
                
                lbl='Closed'
            break

        if(rpred[0]==0 and lpred[0]==0):
            score=score+1
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
        else:
           score=score-1
           cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
        if(score<0):
           score=0   
           cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        if(score>=1):
        #person is feeling sleepy so we beep the alarm
             cv2.imwrite(os.path.join(path,'image.jpg'),frame)
             try:
                # sound.play()
                #  winsound.Beep(2500, 2000)
                winsound.Beep(2500,200)
            
             except:  # isplaying = False
                  pass
             if(thicc<16):
                 thicc= thicc+2
             else:
                 thicc=thicc-2
                 if(thicc<2):
                    thicc=2
             cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 






        # success, image = self.video.read()
        # image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        # gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        # for (x,y,w,h) in face_rects:
        #  cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        #  break
        # ret, jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()

        # success, image = self.video.read(0)
        # height,width = image.shape[0:2]
        # image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        # gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # faces=face_cascade.detectMultiScale(gray,1.3,5)
        # eyes= eye_cascade.detectMultiScale(gray, 1.3,5)
        # cv2.rectangle(image, (0,height-50),(200,height),(0,0,0),thickness=cv2.FILLED) 
        # for (x,y,w,h) in faces:
        #   cv2.rectangle(image,pt1=(x,y),pt2=(x+w,y+h), color= (255,0,0), thickness=3 )

        #   for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(image,pt1=(ex,ey),pt2=(ex+ew,ey+eh), color= (0,0,0), thickness=2 )
             # preprocessing steps
            # eye= image[ey:ey+eh,ex:ex+ew]
            # eye= cv2.resize(eye,(80,80))
            # eye= eye/255
            # eye= eye.reshape(80,80,3)

            #  eye= cv2.resize(eye,(24,24))
            #  eye= eye/255
            #  eye= eye.reshape(24,24,-1)
            # eye= np.expand_dims(eye,axis=0)
             # preprocessing is done now model prediction
            # prediction = model.predict(eye)

            #   if eyes are closed
            # if prediction[0][0]>0.30:
            #    cv2.putText(image,'closed',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
            #            thickness=1,lineType=cv2.LINE_AA)
            #    cv2.putText(image,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
            #            thickness=1,lineType=cv2.LINE_AA)
            #    Score=Score+1
            #    if(Score>1):
            #         try:
            #            winsound.Beep(2500, 2000)
            #            if(Score>=20):
            #             Score =20
            #            # sound.play()
                      
            #         except:
            #           pass

             # if eyes are open
            # elif prediction[0][1]>0.90:
                #   cv2.putText(image,'open',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                #        thickness=1,lineType=cv2.LINE_AA)      
                #   cv2.putText(image,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                #        thickness=1,lineType=cv2.LINE_AA)
                 
                #   Score = Score-1
                #   if (Score<0):
                #     Score=0




        #   print(Score)
          # break

    #     cv2.imshow('frame',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
        print(score)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()