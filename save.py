import cv2

cap = cv2.VideoCapture(0)

frame_Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

codec = cv2.VideoWriter_fourcc(*'XVID')
recorder = cv2.VideoWriter("myvideo.avi",codec,30,(frame_Width,frame_Height))

while True:
    ret,frame = cap.read()
    if not ret:
        print("Could not load the webcam")
        break

    recorder.write(frame)
    cv2.imshow("Webcam feed",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting ...")
        break

cap.release()
recorder.release()
cv2.destroyAllWindows()