
# coding: utf-8

# In[3]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")

img=cv2.imread("image.jpg")
#plt.imshow(img,cmap="gray")
#plt.show()
cap=cv2.VideoCapture(0)


# In[5]:


"""Now we find the faces in the image. 
If faces are found, it returns the positions of detected faces as Rect(x,y,w,h). 
Once we get these locations, we can create a ROI for the face and apply eye detection on this 
ROI (since eyes are always on the face !!! ).
"""
#ROI: A region of interest (ROI) is a portion of an image that you want to filter or perform some other operation on.
#In ROI we will get eyes
cv2.imshow("Colored Img",img)
#cv2.imshow("Colored Image",img)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image",img_gray)
faces=face_cascade.detectMultiScale(img_gray,1.3,5)

#Getting the position of detected face as a rectangle
for (x,y,w,h) in faces:

    #img,initial points,width and height of rect,color of rect,thichkness of rect
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray=img_gray[y:y+h,x:x+w]
    roi_color=img[y:y+h,x:x+w]
    
    eyes=eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#cv2.imshow("image",img)
#cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow("Output Image",img)
cv2.imwrite("outputimage.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


#Similary We Can Apply It On WebCam
"""Now we find the faces in the image. 
If faces are found, it returns the positions of detected faces as Rect(x,y,w,h). 
Once we get these locations, we can create a ROI for the face and apply eye detection on this 
ROI (since eyes are always on the face !!! ).
"""
#ROI: A region of interest (ROI) is a portion of an image that you want to filter or perform some other operation on.
i=1
name="surprise"
while 1:
    ret,img=cap.read()
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(img_gray)
    for (x,y,w,h) in faces:
        #img,initial points,width and height of rect,color of rect,thichkness
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
        roi_gray=img_gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
    cv2.imshow("image",img)
    key=cv2.waitKey(30) & 0xff
    if key==27:    #esc key
       
        break
    if key==ord("s"):
        name=name + str(i)
        name2=name+".jpg"
        cv2.imwrite(name2,roi_gray)
    i+=1

cap.release()
cv2.destroyAllWindows()
    


# In[ ]:




