import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

def send_alert_email(date, location, image_path):
    sender_email = 'gutboiz01@gmail.com'
    sender_password = 'rqdk mird ealw kpeu'
    receiver_email = "nguyenphuccuongtm@gmail.com"

    # Tạo email
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = '📸 Phát hiện tai nạn giao thông!'

    # Nội dung thư (f-string)
    body = f'🚨 Phát hiện tai nạn giao thông tại {location}. Thời gian: {date}'
    message.attach(MIMEText(body, 'plain'))

    # Đính kèm ảnh
    with open(image_path, 'rb') as img_file:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(img_file.read())
        encoders.encode_base64(part)
        filename = os.path.basename(image_path)
        part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
        message.attach(part)

    # Gửi email
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(message)
        print("✅ Email đã được gửi thành công.")
