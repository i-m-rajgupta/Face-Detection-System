import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = "./face_dataset/"

# create folder if it doesn't exist
os.makedirs(dataset_path, exist_ok=True)

file_name = input("Enter the name of person : ")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]

        offset = 5
        y1 = max(0, y-offset)
        y2 = y+h+offset
        x1 = max(0, x-offset)
        x2 = x+w+offset

        face_offset = frame[y1:y2, x1:x2]
        face_selection = cv2.resize(face_offset, (100,100))

        skip += 1
        if skip % 10 == 0:
            face_data.append(face_selection)
            print("Captured:", len(face_data))

        cv2.imshow("Face", face_selection)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Camera", frame)

    # Check for 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting program...")
        break

    # Stop automatically after collecting 100 face images
    if len(face_data) >= 100:
        print("Collected 100 face images. Exiting...")
        break

# Convert face data to numpy array and save
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

np.save(dataset_path + file_name, face_data)
print("Dataset saved at:", dataset_path + file_name + ".npy")

cap.release()
cv2.destroyAllWindows()