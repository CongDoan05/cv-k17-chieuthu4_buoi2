import cv2 as cv
import numpy as np
import os
import smtplib
from email.mime.text import MIMEText

recog_tool = cv.face.LBPHFaceRecognizer_create()
recog_tool.read("face_recognizer_model.yml") #đọc mô hình đã huấn luyện
label_dict = np.load("label_dict.npy", allow_pickle=True).item()
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")   

# Đường dẫn đến thư mục chứa dữ liệu
dataset_path = "data"

AUTHORIZED_USERS = [
    name for name in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, name))
]
print("Danh sach user:", AUTHORIZED_USERS)

# Hàm gửi email thông báo khi mở khóa thành công
def send_email_notification(user):
    try:
        msg = MIMEText(f"User {user} đã được mở khóa.")
        msg["Subject"] = "Access Granted"
        msg["From"] = "nguyendanghuuu1234@gmail.com"
        msg["To"] = "doankukich123@gmail.com"

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.starttls()
            server.login("nguyendanghuuu1234@gmail.com", "ljxqksigaevqxxaf")
            server.send_message(msg)

        print("Email da gui")

    except Exception as e:
        print("Loi gui email:", e)
# Biến để theo dõi trạng thái đã gửi email hay chưa
email_sent = False


cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Loi camera!")
        break
    if frame is not None:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
        dgray = cv.GaussianBlur(gray, (5, 5), 0)
        faces = face_cascade.detectMultiScale(dgray, 1.3, 5)
        
        
        for (x, y, w, h) in faces:
            face_img = dgray[y:y+h, x:x+w]
            name, dotincay = recog_tool.predict(face_img)
            if dotincay < 80: #ngưỡng để xác định có nhận diện
                name = label_dict[name]
                color = (0, 255, 0)
                # Chỉ gửi email khi người được nhận diện là người có trong danh sách và chưa gửi email trước đó
                if name in AUTHORIZED_USERS and not email_sent:
                    print("Mo khoa:", name)
                    send_email_notification(name)
                    email_sent = True
            else:
                name = "Unknown"
                color = (0, 0, 255)
            cv.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv.putText(frame, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv.imshow("Face Recognition", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()