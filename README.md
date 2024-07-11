# AI OpenCV 영상 처리 
이 코드는 얼굴을 감지하고 얼굴 간의 거리를 측정하여 일정 거리 내로 접근했을 때 경고음을 재생하는 프로그램입니다. 주요 구성 요소별로 설명하겠습니다
def text_to_speech(text, lang='ko'):
    tts = gTTS(text=text, lang=lang)
    filename = 'temp.mp3'
    tts.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(1)

    pygame.mixer.quit()
    os.remove(filename)
이 함수는 주어진 텍스트를 음성으로 변환하고 이를 재생하는 기능을 합니다.

1.gTTS를 사용하여 텍스트를 음성으로 변환.


2.변환된 음성을 임시 파일(temp.mp3)로 저장.


3.pygame.mixer를 사용하여 음성을 재생.


4.음성이 재생되는 동안 대기하고, 재생이 끝나면 리소스를 정리하고 임시 파일을 삭제.

#이 함수는 프레임 내에서 얼굴을 감지하는 역할을 합니다.

def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


1. 컬러 이미지를 그레이스케일로 변환.


2. detectMultiScale 함수를 사용하여 얼굴을 감지하고, 감지된 얼굴 좌표를 반환.

