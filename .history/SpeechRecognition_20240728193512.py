import speech_recognition as sr

# 打印库的版本
print(sr.__version__)

# 创建识别器对象
r = sr.Recognizer()

# 尝试读取音频文件
try:
    with sr.AudioFile('此处输入自己的音频路径') as source:
        audio = r.record(source)  # 这里可能不需要 record 方法
        # 使用 Google 识别音频，指定语言为简中
        result = r.recognize_google(audio, language='zh-CN')
        print("Google Speech Recognition thinks you said: " + result)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand the audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")