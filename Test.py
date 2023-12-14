import cv2
import streamlink
import tensorflow as tf
import numpy as np
import time

def preprocess_image(frame):
    resized_frame = cv2.resize(frame, (640, 640))
    resized_frame = resized_frame / 255.0
    return np.expand_dims(resized_frame, axis=0)

def process_prediction(prediction, confidence_threshold=0.5):
    boxes, confidences, class_ids = [], [], []
    for output in prediction:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x, center_y, width, height = (detection[:4] * np.array([640, 640, 640, 640])).astype('int')
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

def draw_predictions(frame, boxes, confidences, class_ids):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = f"Object: {class_ids[i]} - Class: {class_ids[i]}"
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

model = tf.keras.models.load_model('saved_model/yolov8s_float32.h5')

streams = streamlink.streams('https://eofficev2.bekasikota.go.id/backupcctv/m3/bendungan_prisdo.m3u8')
stream_url = streams['best'].url
cap = cv2.VideoCapture(stream_url)

time_between_frames = 1 / 24
last_frame_time = time.time()

while True:
    ret, frame = cap.read()
    if ret:
        current_time = time.time()
        if current_time - last_frame_time >= time_between_frames:
            fps = 1 / (current_time - last_frame_time)
            last_frame_time = current_time
            processed_frame = preprocess_image(frame)
            prediction = model.predict(processed_frame)

            boxes, confidences, class_ids = process_prediction(prediction)
            draw_predictions(frame, boxes, confidences, class_ids)
            
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()