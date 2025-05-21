import cv2
import time
import threading
import numpy as np
from kafka import KafkaProducer
import config
import json
import base64


producer = KafkaProducer(
    bootstrap_servers=[config.kafka_ip],
    max_request_size=9000000,
)

topic_name = "input"

def send_video(video_path, stream_id):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize để đồng nhất đầu vào cho model
        resized_frame = cv2.resize(frame, (250, 250))
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Encode thành ảnh byte
        _, buffer = cv2.imencode('.jpg', frame_rgb)
        frame_bytes = base64.b64encode(buffer).decode('utf-8')

        message = {
            'source_id': stream_id,
            'frame': frame_bytes
        }

        producer.send(topic_name, json.dumps(message).encode('utf-8'))
        producer.flush()

        cv2.imshow(f"Stream {stream_id}", cv2.resize(frame, (500, 500)))
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

        time.sleep(0.03)

    cap.release()
    cv2.destroyWindow(f"Stream {stream_id}")


if __name__ == "__main__":
    video_paths = [
        "D:/HAUI/DATN/Accident_detection/Code1/Accident-Detection-System/video/test.mp4",
        "D:/HAUI/DATN/Accident_detection/Code1/Accident-Detection-System/video/test1.mp4"
    ]

    threads = []
    for i, path in enumerate(video_paths):
        t = threading.Thread(target=send_video, args=(path, str(i + 1)))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
