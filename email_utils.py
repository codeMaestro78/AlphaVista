import smtplib
from email.mime.text import MIMEText

def send_email_alert(to_email, subject, message, from_email, from_password, smtp_server='smtp.gmail.com', smtp_port=587):
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, [to_email], msg.as_string()) 