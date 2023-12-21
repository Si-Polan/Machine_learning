from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import streamlink
import cv2
import imutils
from imutils.contours import sort_contours
from datetime import datetime
import requests
import json

url_api = 'URL_API'
url_api_backup = 'URL_API_BACKUP'

vehicle_model = YOLO("./models/vehicle.pt")
helmet_model = YOLO("./models/helmet.pt")
plate_model = YOLO("./models/plate.pt")
seatbelt_model = YOLO("./models/seatbelt.pt")
ocr_model = tf.keras.models.load_model("./models/ocr.h5")

streams = streamlink.streams('https://cctvjss.jogjakota.go.id/atcs/ATCS_Simpang_Tugu.stream/chunklist_w857074814.m3u8')
stream_url = streams['best'].url
cap = cv2.VideoCapture(stream_url)

def ocr_plate(number_plate_image):
    gray = cv2.cvtColor(number_plate_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 50, 200)
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sort_contours(contours, method="left-to-right")[0]

    characters = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if (10 <= w <= 200) and (20 <= h <= 150):
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            if tW > tH:
                thresh = imutils.resize(thresh, width=32)
            else:
                thresh = imutils.resize(thresh, height=32)

            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)

            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))

            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)

            characters.append((padded, (x, y, w, h)))

    boxes = [b[1] for b in characters]
    characters = np.array([c[0] for c in characters], dtype="float32")
    preds = ocr_model.predict(characters)

    label_names = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_names = [l for l in label_names]

    output = ""
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        i = np.argmax(pred)
        label = label_names[i]
        output += label
    return output

def send_data(vehicle_image, type):
    number_plate_results = plate_model.predict(vehicle_image)
    for npr in number_plate_results:
        for np_box in npr.boxes:
            if plate_model.names[int(np_box.cls)] == "platNomor":
                np_x1, np_y1, np_x2, np_y2 = np_box.xyxy[0]
                number_plate_image = vehicle_image[int(np_y1):int(np_y2), int(np_x1):int(np_x2)]
                plate = ocr_plate(number_plate_image)
                now = datetime.now()

                _, img_encoded = cv2.imencode('.jpg', vehicle_image)
                image_data = img_encoded.tobytes()
                data = {
                    "location": "Simpang Tugu Yogyakarta",
                    "type": type,
                    "vehicle_number_plate": plate,
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S")}
                files = {
                    'image': ('image.jpg', image_data, 'image/jpeg'),
                    'data': (None, json.dumps(data), 'application/json')}
                headers = {'Content-Type': 'multipart/form-data'}
                response = requests.post(url_api, files=files, headers=headers)
                if response.status_code != 200:
                    requests.post(url_api_backup, files=files, headers=headers)
    return

while True:
    ret, frame = cap.read()
    image = cv2.resize(frame, (1280, 1280))
    vehicle_result = vehicle_model.predict(image)

    for r in vehicle_result:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            track_ids = box.id
            vehicle_image = image[int(y1):int(y2), int(x1):int(x2)]
            vehicle_image = cv2.resize(vehicle_image, (640, 640))

            if vehicle_model.names[int(box.cls)] == "bike":
                violation_results = helmet_model.predict(vehicle_image)
                for vr in violation_results:
                    for v_bo in vr.boxes:
                        if helmet_model.names[int(v_bo.cls)] == "Without Helmet":
                            send_data(vehicle_image, "Tidak Memakai Helm")

            elif vehicle_model.names[int(box.cls)] == "car":
                violation_results = seatbelt_model.predict(vehicle_image)
                for vr in violation_results:
                    for v_bo in vr.boxes:
                        if seatbelt_model.names[int(v_bo.cls)] == "no-seatbelt":
                            send_data(vehicle_image, "Tidak Memakai Seatbelt")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()