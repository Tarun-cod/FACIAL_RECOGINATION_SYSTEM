import cv2
import numpy as np
import sqlite3

faceDetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)

def insertorupdate(Id,Name,age):
    conn=sqlite3.connect("testdata.db")
    cmd="SELECT * FROM STUDENTS WHERE Id="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        conn.execute("UPDATE STUDENTS SET name=?,age=? WHERE id=?",(Name,age,Id))
        conn.execute("UPDATE STUDENTS SET age=?,age=? WHERE id=?",(Name,age,Id))
    else:
        conn.execute("INSERT INTO STUDENTS (Id,Name,age) values(?,?,?)",(Id,Name,age))
    conn.commit()
    conn.close()

Id=input("ENTER USER ID: ")
Name=input("ENTER NAME: ")
age=input("ENTER AGE: ")

insertorupdate(Id,Name,age)

sampleNum=0
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1
        cv2.imwrite("dataset/User." + str(Id) + "." + str(sampleNum) + ".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(60)
    cv2.imshow("Face",img)
    cv2.waitKey(1)
    if(sampleNum>100):
        break

cam.release()
cv2.destroyAllWindows()

