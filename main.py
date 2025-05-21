import cv2
from detection import AccidentDetectionModel
from email_sender import send_alert_email
import numpy as np
import os
from kafka import KafkaProducer, KafkaConsumer
import config
import json
import base64


model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX
flag = False

topic_name_in = "input"
c = KafkaConsumer(
    topic_name_in,
    bootstrap_servers = [config.kafka_ip],
    auto_offset_reset = 'latest',
    enable_auto_commit = True,
    fetch_max_bytes = 9000000,
    fetch_max_wait_ms = 1000,
)

topic_name_out = "output"
p = KafkaProducer(
    bootstrap_servers = [config.kafka_ip],
    max_request_size = 9000000,
)


def startapplication():
    # Lấy video từ local
    # video = cv2.VideoCapture('D:/HAUI/DATN/Accident_detection/Code4/Accident-Detection-Web-App/model-implementor/assets/car-crash.mp4')
    
    # while True:
    #     ret, frame = video.read()
    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     roi = cv2.resize(gray_frame, (250, 250))

    #     pred, prob = model.predict_accident(roi[np.newaxis, :, :])
    #     if(pred == "Accident"):
    #         prob = (round(prob[0][0]*100, 2))
    #         flag = True
    #         if(prob > 90):
    #             # send_alert_email()
    #             cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
    #             cv2.putText(frame, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)

    #     if cv2.waitKey(33) & 0xFF == ord('q'):
    #         if flag:
    #             send_alert_email()
    #         return
    #     cv2.imshow('Video', frame)  


## tốc độ video nhanh
    # for message in c:
    #     stream = message.value
    #     # Chuyển thành hình ảnh
    #     stream = np.frombuffer(stream, dtype=np.uint8)
    #     image = cv2.imdecode(stream, cv2.IMREAD_COLOR)

    #     pred, prob = model.predict_accident(image[np.newaxis, :, :])
    #     if(pred == "Accident"):
    #         prob = (round(prob[0][0]*100, 2))
    #         if(prob > 90):
    #             # send_alert_email()
    #             cv2.rectangle(image, (0, 0), (280, 40), (0, 0, 0), -1)
    #             cv2.putText(image, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)
    #             print("Phát hiện tai nạn giao thông")
    #             ret, buffer = cv2.imencode('.jpg', image)
    #             p.send(topic_name_out, buffer.tobytes())
    #             p.flush()
    # return



    for message in c:
        data = json.loads(message.value.decode('utf-8'))
        frame_data = base64.b64decode(data['frame'])

        #chuyển bytes thành ảnh
        stream = np.frombuffer(frame_data, dtype=np.uint8)
        image = cv2.imdecode(stream, cv2.IMREAD_COLOR)

        pred, prob = model.predict_accident(image[np.newaxis, :, :])
        if(pred == "Accident"):
            prob = (round(prob[0][0]*100, 2))
            if(prob > 90):
                stream_id = data['source_id']
                # send_alert_email()
                cv2.rectangle(image, (0, 0), (280, 40), (0, 0, 0), -1)
                cv2.putText(image, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)
                stream_id = int(stream_id) + 1
                print(f"Phát hiện tai nạn giao thông tại stream {stream_id}")
                ret, buffer = cv2.imencode('.jpg', image)
                p.send(topic_name_out, buffer.tobytes())
                p.flush()
    return

if __name__ == '__main__':
    startapplication()