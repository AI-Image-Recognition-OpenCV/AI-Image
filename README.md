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
