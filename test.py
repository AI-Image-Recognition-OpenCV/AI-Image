import pyttsx3

def texttospeech(text):
    # pyttsx3 엔진 초기화
    engine = pyttsx3.init()

    # 속성 설정 (속도, 볼륨, 목소리 등)
    rate = engine.getProperty('rate')   # 기본 속도 가져오기
    engine.setProperty('rate', rate - 50)  # 속도 줄이기
    volume = engine.getProperty('volume')  # 기본 볼륨 가져오기
    engine.setProperty('volume', volume + 0.25)  # 볼륨 높이기

    # 한국어 음성 설정 (시스템에 따라 다를 수 있음)
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'ko' in voice.id:  # 한국어 음성 찾기
            engine.setProperty('voice', voice.id)
            break

    # 텍스트를 음성으로 변환
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    text = "노란선 안으로 들어가세요."
    texttospeech(text)
