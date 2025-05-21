kafka_ip = "127.0.0.1:9092"


# from kafka import KafkaConsumer
# import cv2
# import numpy as np

# # Tạo consumer để lắng nghe topic output
# consumer = KafkaConsumer(
#     'output',
#     bootstrap_servers='127.0.0.1:9092',
#     auto_offset_reset='latest',
#     value_deserializer=lambda x: x  # giữ nguyên bytes
# )

# for msg in consumer:
#     # msg.value là bytes → chuyển thành numpy array
#     img_array = np.frombuffer(msg.value, dtype=np.uint8)

#     # Giải mã thành ảnh (OpenCV image)
#     frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#     if frame is not None:
#         # Hiển thị ảnh
#         cv2.imshow("Kafka Streamed Frame", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         print("Không giải mã được ảnh!")

# cv2.destroyAllWindows()
