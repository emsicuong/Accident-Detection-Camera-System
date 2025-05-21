import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_alert_email():
    sender_email = 'gutboiz01@gmail.com'
    sender_password = 'rqdk mird ealw kpeu'

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = "nguyenphuccuongtm@gmail.com"
    message['Subject'] = 'Accident Detected!'
    body = 'An accident has been detected in the uploaded video. Please check immediately.'
    message.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.send_message(message)
    server.quit()
