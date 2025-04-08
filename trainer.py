import os
import cv2
import numpy as np
from PIL import Image


recognizer = cv2.face.LBPHFaceRecognizer_create()
path= "dataset"

def get_images_with_id(path):
    images_path=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for image_path in images_path:
        face_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if face_img is None:
            print(f"Skipping file {image_path} as it could not be read")
            continue
        face_np=np.array(face_img,'uint8')
        id=int(os.path.split(image_path)[-1].split('.')[1])
        faces.append(face_np)
        print(id)
        ids.append(id)
        cv2.imshow("training",face_np)
        cv2.waitKey(10)

    return np.array(ids),faces

ids,faces=get_images_with_id(path)
if len(ids) > 1 and len(faces) > 1:
    recognizer.train(faces, ids)
    recognizer.save("recognizer/trainingData.yml")
    print("Training complete and data saved.")
else:
    print("Not enough data to train. Ensure you have at least two images with valid IDs.")
recognizer.save("recognizer/trainingData.yml")
cv2.destroyAllWindows()