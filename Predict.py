from ultralytics import YOLO
import numpy as np
import streamlink
import cv2
import easyocr

helmet_model = YOLO("./Bike-Helmet-Detection-1/runs/detect/train/weights/best.pt")
person_bike_model = YOLO("./detect-person-on-motorbike-or-scooter/runs/detect/train/weights/best.pt")
number_plate_model = YOLO("./Vehicle-Registration-Plates-1/runs/detect/train/weights/best.pt")

streams = streamlink.streams('https://cctvjss.jogjakota.go.id/atcs/ATCS_Simpang_Tugu.stream/chunklist_w857074814.m3u8')
stream_url = streams['best'].url
cap = cv2.VideoCapture(stream_url)

reader = easyocr.Reader(['en', 'en'])

while True:
    ret, frame = cap.read()
    person_bike_results = person_bike_model.track(frame, show=True)

    for r in person_bike_results:
        for box in r.boxes:
            if person_bike_model.names[int(box.cls)] == "Person_Bike":
                x1, y1, x2, y2 = box.xyxy[0]
                track_ids = box.id
                print(track_ids)
                person_bike_image = frame[int(y1):int(y2), int(x1):int(x2)]
                helmet_results = helmet_model.predict(person_bike_image)

                for hr in helmet_results:
                    for h_bo in hr.boxes:
                        if helmet_model.names[int(h_bo.cls)] == "Without_Helmet" :
                            number_plate_results = number_plate_model.predict(person_bike_image)

                            for npr in number_plate_results:
                                for np_box in npr.boxes:
                                    if number_plate_model.names[int(np_box.cls)] == "License_Plate":
                                        np_x1, np_y1, np_x2, np_y2 = np_box.xyxy[0]
                                        number_plate_image = person_bike_image[int(np_y1):int(np_y2), int(np_x1):int(np_x2)]

                                        gray = cv2.cvtColor(number_plate_image, cv2.COLOR_RGB2GRAY)
                                        result = reader.readtext(gray)
                                        print(result[0][1])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()