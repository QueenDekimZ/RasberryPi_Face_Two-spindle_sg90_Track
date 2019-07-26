import RPi.GPIO as GPIO
from time import sleep
import face_recognition
import cv2

btm_kp = 10 # 底部舵机的Kp系数
top_kp = 10 # 顶部舵机的Kp系数

last_btm_degree = 50 # 最近一次底部舵机的角度值记录
last_top_degree = 50 # 最近一次顶部舵机的角度值记录

offset_dead_block = 0.1 # 设置偏移量的死区

def setServoAngle(servo, angle):
    #assert angle >= 15 and angle <= 115
    pwm = GPIO.PWM(servo, 50)
    pwm.start(8)
    dutyCycle = angle / 10. + 3
    pwm.ChangeDutyCycle(dutyCycle)
    sleep(0.1)
    pwm.stop()

def btm_servo_control(offset_x):
    global offset_dead_block #threshold
    global btm_kp   #linear_angle_rate
    global last_btm_degree 
    
    if abs(offset_x) < offset_dead_block:
        offset_x = 0
    offset_x *= -1
    delta_degree = offset_x * btm_kp
    
    next_btm_degree = last_btm_degree + delta_degree
     # 添加边界检测
    if next_btm_degree < 0:
        next_btm_degree = 0
    elif next_btm_degree > 180:
        next_btm_degree = 180

    return int(next_btm_degree)

def top_servo_control(offset_y):
    '''
    顶部舵机的比例控制
    这里舵机使用开环控制
    '''
    global offset_dead_block
    global top_kp # 控制舵机旋转的比例系数
    global last_top_degree # 上一次顶部舵机的角度

    # 如果偏移量小于阈值就不相应
    if abs(offset_y) < offset_dead_block:
        offset_y = 0

    offset_y *= -1
    # offset范围在-50到50左右
    delta_degree = offset_y * top_kp
    # 新的顶部舵机角度
    next_top_degree = last_top_degree + delta_degree
    # 添加边界检测
    if next_top_degree < 15:
        next_top_degree = 15
    elif next_top_degree > 115:
        next_top_degree = 115
    
    return int(next_top_degree)

def calculate_offset(img_width, img_height, face):
    '''
    计算人脸在画面中的偏移量
    偏移量的取值范围： [-1, 1]
    '''
    #top right bottom left
    (y, r, b, x) = face
    w = r - x
    h = b - y
    #(x, y, w, h) = face
    face_x = float(x + w/2.0)
    face_y = float(y + h/2.0)
    # 人脸在画面中心X轴上的偏移量
    offset_x = (float(face_x / img_width - 0.5) * 2 + 0.75)*4
    # 人脸在画面中心Y轴上的偏移量
    offset_y = (float(face_y / img_height - 0.5) * 2 + 0.75)*4

    return (offset_x, offset_y)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

row = 13
col = 19

GPIO.setup(row, GPIO.OUT)
GPIO.setup(col, GPIO.OUT)
# 舵机角度初始化
setServoAngle(row, last_btm_degree)
print("Servo init")
setServoAngle(col, last_top_degree)

#ip_camera_url = 'http://192.168.1.10:8080/?action=stream'
#ip_camera_url = 0
video_capture = cv2.VideoCapture(0)
# 设置缓存区的大小 !!!
video_capture.set(cv2.CAP_PROP_FPS, 30)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE,10)


print('IP摄像头是否开启： {}'.format(video_capture.isOpened()))

obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]

known_face_names = [
    "Barack Obama",
    "Joe Biden"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame= video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
	
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            face_names.append(name)
    process_this_frame = not process_this_frame
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255) , 1)
    if face_names:
        
        img_height, img_width, _ = frame.shape
        (offset_x, offset_y) = calculate_offset(img_width, img_height, face_locations[0]) 
        #calcuate next rotate angle
        next_btm_degree = btm_servo_control(offset_x)
        next_top_degree = top_servo_control(offset_y)
        setServoAngle(row, next_btm_degree)
        setServoAngle(col, next_top_degree)
        
        last_btm_degree = next_btm_degree
        last_top_degree = next_top_degree
        print("X轴偏移量：{} Y轴偏移量：{}".format(offset_x, offset_y))
        print('底部角度： {} 顶部角度：{}'.format(next_btm_degree, next_top_degree))
    cv2.imshow('FaceDetect', frame)
    if cv2.waitKey(1)  == ord('q'):
        break
    elif cv2.waitKey(1) == ord('r'):
        print('舵机重置')
        # 重置舵机
        # 最近一次底部舵机的角度值记录
        last_btm_degree = 50
        # 最近一次顶部舵机的角度值记录
        last_top_degree = 50
        # 舵机角度初始化
        setServoAngle(row, last_btm_degree)
        setServoAngle(col, last_top_degree)

GPIO.cleanup()
video_capture.release()
cv2.destroyAllWindows()
