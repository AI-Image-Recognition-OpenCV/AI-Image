<img src = file:///C:/Users/User/Downloads/opencv-color.svg>
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


