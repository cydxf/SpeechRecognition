import speech_recognition as sr
import moviepy.editor as mp
from pydub import AudioSegment
import os

def extract_audio(file_path):
    """
    从输入文件中提取音频，支持 wav、mp4 和 mp3 文件。

    参数:
    file_path (str): 输入文件路径

    返回:
    str: 提取后的音频文件路径
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    audio_path = "temp_audio.wav"

    if file_extension == ".wav":
        return file_path
    elif file_extension in [".mp4", ".mp3"]:
        if file_extension == ".mp4":
            video = mp.VideoFileClip(file_path)
            video.audio.write_audiofile(audio_path)
        elif file_extension == ".mp3":
            audio = AudioSegment.from_mp3(file_path)
            audio.export(audio_path, format="wav")
        return audio_path
    else:
        raise ValueError("不支持的文件格式。请提供 wav、mp4 或 mp3 文件。")

def transcribe_audio(file_path):
    """
    将音频文件转换为文本。

    参数:
    file_path (str): 音频文件路径

    返回:
    str: 转换后的文本
    """
    recognizer = sr.Recognizer()
    audio_file = extract_audio(file_path)

    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language='zh-CN')
            return text
        except sr.UnknownValueError:
            return "Google 语音识别无法理解音频"
        except sr.RequestError as e:
            return f"无法从 Google 语音识别服务请求结果; {e}"

def save_transcription(text, output_file):
    """
    将转换后的文本保存到文件。

    参数:
    text (str): 转换后的文本
    output_file (str): 输出文件路径
    """
    with open(output_file, "w") as file:
        file.write(text)

def main(input_file, output_file):
    """
    主函数，处理输入文件并保存转换结果。

    参数:
    input_file (str): 输入文件路径
    output_file (str): 输出文件路径
    """
    transcription = transcribe_audio(input_file)
    save_transcription(transcription, output_file)
    print(f"转换结果已保存到 {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="将 wav、mp4 和 mp3 文件中的音频转换为文本。")
    parser.add_argument("input_file", type=str, help="输入音频或视频文件路径。")
    parser.add_argument("output_file", type=str, help="输出文本文件路径。")

    args = parser.parse_args()

    main(args.input_file, args.output_file)
